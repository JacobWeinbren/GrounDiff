"""Rasterise scenes into per-channel .npy files (memory-mappable).

One scene = one point-cloud tile plus a target DTM for the same area. The
tile is either classified by lasground_new with default settings
(--before-dir: every point 1 or 2; gives the before -> after model its
lasground_new inputs) or used as points only (--points-dir: any tile, e.g.
the published EA files as downloaded; classes ignored; for the DSM -> DTM
model, which needs no LAStools). The target is either

  * the EA's published DTM raster (--dtm-dir: GeoTIFF/ASCII grids/VRTs in any
    tiling; the EA builds it from its hand-edited ground class), or
  * a hand-edited copy of the tile (--after-dir: e.g. your own LP360 output;
    the TIN of its ground class 2 is the target).

The classes in the published EA LAZ/COPC files are never used: they come
from a different process and do not match the DTM rasters. Run lasground_new
on the published tiles first (groundiff.data.lasground).

    python -m groundiff.data.preprocess --before-dir before/ --dtm-dir ea_dtm/ \\
        --out data/scenes_1m --gsd 1.0 --workers 6
    python -m groundiff.data.preprocess --points-dir data/laz/ea --dtm-dir ea_dtm/ \\
        --out data/scenes_1m --gsd 1.0 --workers 6          # no LAStools needed

Output layout:
    <out>/<scene>/meta.json
    <out>/<scene>/<channel>.npy          float32 [H, W], NaN = no data
Channels: dsm_max, dsm_min, dsm_last, density, z_std, echoes, has_return,
in_survey, dtm_before (+ before_valid), sem_ground, sem_nonground, gt_dtm
(+ gt_valid), and with --after-dir also top_ground.

Quality gate: a scene whose target disagrees with the points (different
survey, misregistration, wrong product) is marked "suspect" in meta.json and
skipped by training. Measured on the cells lasground_new calls ground, or,
for points-only scenes, on open ground (single returns, height spread
< 5 cm), where the lowest return should sit on the DTM.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .laz import class_histogram, read_points
from .rasterise import Grid, input_rasters, target_from_points, target_from_rasters

SCHEMA = 3
LAZ_SUFFIXES = (".laz", ".las")
RASTER_SUFFIXES = (".tif", ".tiff", ".asc", ".vrt", ".img")
LASGROUND_CLASSES = {1, 2, 7, 18}


def scene_name(path: Path) -> str:
    name = Path(path).name
    for suf in (".copc.laz", ".laz", ".las"):
        if name.lower().endswith(suf):
            return name[: -len(suf)]
    return Path(path).stem


def _default_crs():
    """EA open-data COPC tiles carry no CRS VLR; they are British National Grid."""
    from ..io_raster import crs_wkt_from_epsg
    return crs_wkt_from_epsg(27700)


def _save(out: Path, name: str, arr: np.ndarray):
    np.save(out / f"{name}.npy", np.ascontiguousarray(arr, dtype=np.float32))


def _stamp(p: Path | None) -> list | None:
    if p is None:
        return None
    st = Path(p).stat()
    return [str(p), st.st_size, int(st.st_mtime)]


def check_lasground_classes(hist: dict, max_other: float = 0.005) -> str | None:
    """None if the classes look like lasground_new output, else a message."""
    hist = {int(k): v for k, v in hist.items()}
    total = sum(hist.values()) or 1
    other = {k: v for k, v in hist.items() if k not in LASGROUND_CLASSES}
    share = sum(other.values()) / total
    if share > max_other:
        return (f"{share:.1%} of points have classes {sorted(other)}; lasground_new (default settings) "
                "writes only 1/2. Is this the published EA file rather than lasground_new output?")
    if not hist.get(2):
        return "no ground (class 2) points"
    return None


def quality(arrs: dict, alpha: float = 0.2) -> dict:
    """Agreement of the target with lasground_new where lasground_new says
    ground: most such cells need no edit, so a low share means the target
    does not belong to this point cloud."""
    q = {}
    ok = np.isfinite(arrs["gt_dtm"]) & (np.nan_to_num(arrs["gt_valid"]) > 0.5)
    survey = np.nan_to_num(arrs["in_survey"]) > 0.5
    q["target_coverage"] = float(ok[survey].mean()) if survey.any() else 0.0
    if "dtm_before" in arrs:
        g = ok & (np.nan_to_num(arrs["sem_ground"]) > 0.5) & np.isfinite(arrs["dtm_before"])
        ref = arrs["dtm_before"]
    else:                                    # points only: open ground, lowest return ~ ground
        g = (ok & np.isfinite(arrs["dsm_min"]) & (np.nan_to_num(arrs["z_std"], nan=9.0) < 0.05)
             & (np.nan_to_num(arrs.get("echoes", np.ones_like(arrs["dsm_min"])), nan=9.0) <= 1.05))
        ref = arrs["dsm_min"]
    # a DSM given as the target would follow canopy/roof tops where points have several echoes
    tall = (ok & np.isfinite(arrs["dsm_max"]) & (np.nan_to_num(arrs.get("echoes", np.zeros(1)), nan=0) >= 1.5)
            & (np.nan_to_num(arrs["z_std"], nan=0) > 1.0))
    if tall.sum() >= 200:
        q["canopy_gap_m"] = float(np.median((arrs["dsm_max"] - arrs["gt_dtm"])[tall]))
    q["n_ground_cells"] = int(g.sum())
    if g.sum() >= 50:
        d = (arrs["gt_dtm"] - ref)[g].astype(np.float64)
        q["median_dz"] = float(np.median(d))
        q["agree_frac"] = float((np.abs(d) < alpha).mean())
    return q


def is_suspect(q: dict, min_agree: float = 0.6, max_offset: float = 0.15, min_coverage: float = 0.5) -> list:
    why = []
    if q.get("canopy_gap_m") is not None and q["canopy_gap_m"] < 1.0:
        why.append(f"target follows the tree/building tops (median {q['canopy_gap_m']:.2f} m below the highest "
                   "return where there are several echoes): a DSM, not a DTM?")
    if q.get("target_coverage", 0) < min_coverage:
        why.append(f"target covers {q.get('target_coverage', 0):.0%} of the survey area")
    if "agree_frac" in q and q["agree_frac"] < min_agree:
        why.append(f"only {q['agree_frac']:.0%} of lasground_new ground cells within 0.2 m of the target")
    if "median_dz" in q and abs(q["median_dz"]) > max_offset:
        why.append(f"median offset {q['median_dz']:+.2f} m on lasground_new ground")
    return why


def process_scene(before: Path, out_root: Path, *, dtm_paths: list | None = None, after: Path | None = None,
                  gsd: float = 1.0, ground_classes=(2,), before_ground_classes=(2,), lasground: bool = True,
                  overwrite: bool = False, read_opts: dict | None = None, coverage_close_m: float = 30.0,
                  gate: dict | None = None, geotiff: bool = False) -> dict | None:
    """before: point tile classified by lasground_new (lasground=False: any
    point tile, no dtm_before/sem channels). Target from dtm_paths (rasters)
    or after (hand-edited tile, ground_classes). read_opts: drop_withheld /
    drop_overlap / drop_synthetic / drop_classes for read_points.
    Returns meta, or None when an up-to-date cache exists."""
    before = Path(before)
    read_opts = dict(read_opts or {})
    name = scene_name(before)
    out = Path(out_root) / name
    meta_path = out / "meta.json"
    key = {"schema": SCHEMA, "gsd": gsd, "before": _stamp(before), "after": _stamp(after),
           "dtm": sorted(_stamp(Path(p)) for p in (dtm_paths or [])), "ground_classes": list(ground_classes),
           "before_ground_classes": list(before_ground_classes), "lasground": lasground,
           "read_opts": read_opts, "coverage_close_m": coverage_close_m, "gate": gate or {}}
    key = json.loads(json.dumps(key))
    if meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text())
        if meta.get("cache_key") == key:
            if geotiff:                        # cached scene, GeoTIFFs asked for now: write them from the .npy
                from ..io_raster import write_geotiff
                g = meta["grid"]
                for f in sorted(out.glob("*.npy")):
                    if not f.with_suffix(".tif").exists():
                        write_geotiff(f.with_suffix(".tif"), np.load(f), g["xmin"], g["ymax"], g["gsd"],
                                      meta.get("crs_wkt"))
            return None
    if not dtm_paths and after is None:
        raise ValueError(f"{name}: no target (no DTM raster covers it and no after file)")
    t0 = time.time()

    pts = read_points(before, **read_opts)
    hist = class_histogram(pts.cls)
    if lasground:
        problem = check_lasground_classes(hist)
        if problem:
            raise ValueError(f"{before.name}: {problem}")
    grid = Grid.from_bounds(pts.x.min(), pts.y.min(), pts.x.max(), pts.y.max(), gsd)
    arrs = input_rasters(grid, pts, lasground, before_ground_classes, coverage_close_m)
    survey = arrs["in_survey"] > 0.5
    meta = {"schema": SCHEMA, "scene": name, "gsd": gsd, "grid": grid.to_dict(),
            "crs_wkt": pts.crs_wkt or _default_crs(), "before_file": str(before), "n_points": len(pts),
            "class_hist_before": hist, "has_before": lasground, "lasground": lasground,
            "before_ground_classes": list(before_ground_classes), "read_opts": read_opts}
    if dtm_paths:
        arrs.update(target_from_rasters(grid, [str(p) for p in dtm_paths], survey))
        meta.update({"target": "dtm_raster", "dtm_files": [str(p) for p in dtm_paths]})
    else:
        ap = read_points(after, drop_withheld=read_opts.get("drop_withheld", True))
        arrs.update(target_from_points(grid, ap, survey, ground_classes))
        meta.update({"target": "after_points", "after_file": str(after), "ground_classes": list(ground_classes),
                     "class_hist_after": class_histogram(ap.cls)})
        if len(ap) != len(pts):
            meta["warning"] = f"before/after point counts differ ({len(pts)} vs {len(ap)})"
    q = quality(arrs)
    why = is_suspect(q, **(gate or {}))
    meta["quality"] = {**q, "suspect": bool(why), "reasons": why}
    meta["coverage_close_m"] = coverage_close_m
    out.mkdir(parents=True, exist_ok=True)
    meta_path.unlink(missing_ok=True)          # a failed rerun must not leave the old meta pointing at new files
    for f in list(out.glob("*.npy")) + list(out.glob("*.tif")):
        if f.stem not in arrs:                 # channels from an earlier run with other settings
            f.unlink()
    for k, v in arrs.items():
        _save(out, k, v)
    if geotiff:
        from ..io_raster import write_geotiff
        for k, v in arrs.items():
            write_geotiff(out / f"{k}.tif", v, grid.xmin, grid.ymax, grid.gsd, meta["crs_wkt"])
    meta["cache_key"] = key
    meta["seconds"] = round(time.time() - t0, 1)
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


def find_laz(folder: Path) -> dict[str, Path]:
    files = [p for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in LAZ_SUFFIXES]
    return {scene_name(p): p for p in files}


def index_rasters(folder: Path, cache: Path | None = None) -> list[dict]:
    """Extents of every raster under folder (cached by path, size and mtime)."""
    from ..io_raster import raster_info
    old = {}
    if cache and cache.exists():
        old = {r["path"]: r for r in json.loads(cache.read_text())}
    rows = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() not in RASTER_SUFFIXES or not p.is_file():
            continue
        st = p.stat()
        r = old.get(str(p))
        if not r or r.get("size") != st.st_size or r.get("mtime") != int(st.st_mtime):
            try:
                r = {"path": str(p), "size": st.st_size, "mtime": int(st.st_mtime), **raster_info(p)}
            except Exception as e:
                print(f"[warn] skipping unreadable raster {p}: {e!r}", file=sys.stderr)
                continue
        rows.append(r)
    if cache:
        cache.write_text(json.dumps(rows))
    return rows


PRODUCT_RANK = ("lidar_tiles_dtm", "national_lidar_programme_dtm", "lidar_composite_dtm")


YEAR_IN_PATH = re.compile(r"(?:^|[_/\\-])((?:19|20)\d{2})(?=[_/\\.-])")


def _raster_rank(path: str, year: str | None) -> tuple:
    """Prefer rasters of the point cloud's survey year, then the survey's own
    DTM over the NLP DTM over the mixed-year composite (as groundiff.data.ea_dtm
    names its folders: <product>_<year>_<res>/)."""
    low = path.lower()
    years = set(YEAR_IN_PATH.findall(low))
    year_ok = 0 if (year is None or not years or year in years) else 1
    prod = next((i for i, p in enumerate(PRODUCT_RANK) if p in low), len(PRODUCT_RANK))
    return year_ok, prod


def rasters_for(bounds, index: list[dict], year: str | None = None) -> list[str]:
    """Rasters covering bounds, best first (earlier ones win where they overlap)."""
    x0, y0, x1, y1 = bounds
    hits = [r for r in index if r["xmin"] < x1 and r["xmax"] > x0 and r["ymin"] < y1 and r["ymax"] > y0]
    hits.sort(key=lambda r: _raster_rank(r["path"], year) + (r["res"], r["path"]))
    return [r["path"] for r in hits]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--before-dir", type=Path, help="point tiles classified by lasground_new (default settings)")
    src.add_argument("--points-dir", type=Path, help="point tiles used as points only, classes ignored "
                     "(e.g. the published EA files; for the DSM -> DTM model, no LAStools needed)")
    tg = ap.add_mutually_exclusive_group(required=True)
    tg.add_argument("--dtm-dir", type=Path, help="EA DTM rasters (target)")
    tg.add_argument("--after-dir", type=Path, help="hand-edited tiles with the same names (target = ground class)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gsd", type=float, default=1.0, help="cell size in metres (EA DTM: 1 m)")
    ap.add_argument("--ground-classes", type=int, nargs="+", default=[2], help="target ground classes (--after-dir)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--drop-overlap", action="store_true", help="drop overlap-flagged / class 12 points")
    ap.add_argument("--drop-synthetic", action="store_true", help="drop synthetic-flagged points")
    ap.add_argument("--keep-withheld", action="store_true")
    ap.add_argument("--min-agree", type=float, default=0.6, help="quality gate, see module doc")
    ap.add_argument("--max-offset", type=float, default=0.15, help="quality gate, metres")
    ap.add_argument("--geotiff", action="store_true", help="also write every channel as GeoTIFF")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    before = find_laz(a.before_dir or a.points_dir)
    lasground = a.before_dir is not None
    a.out.mkdir(parents=True, exist_ok=True)
    ro = {"drop_overlap": a.drop_overlap, "drop_synthetic": a.drop_synthetic, "drop_withheld": not a.keep_withheld}
    gate = {"min_agree": a.min_agree, "max_offset": a.max_offset}
    jobs = []
    if a.dtm_dir:
        from .laz import header_info
        index = index_rasters(a.dtm_dir, a.out / "dtm_index.json")
        print(f"{len(index)} DTM rasters under {a.dtm_dir}")
        for k, p in before.items():
            try:
                b = header_info(p)["bounds"]
            except Exception as e:
                print(f"  {p.name}: FAILED reading header {e!r}", file=sys.stderr)
                continue
            from .ea_dtm import survey_year
            hits = rasters_for(b, index, survey_year(p.name))
            if not hits:
                print(f"[warn] {k}: no DTM raster covers it; skipped")
                continue
            jobs.append((p, {"dtm_paths": hits}))
    else:
        after = find_laz(a.after_dir)
        missing = sorted(set(before) - set(after))
        if missing:
            print(f"[warn] {len(missing)} tiles have no after file and are skipped: {missing[:5]}...")
        jobs = [(p, {"after": after[k]}) for k, p in before.items() if k in after]
    print(f"{len(jobs)} scenes -> {a.out}")
    failures, suspect = 0, 0
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futures = {ex.submit(process_scene, p, a.out, gsd=a.gsd, ground_classes=tuple(a.ground_classes),
                             lasground=lasground, overwrite=a.overwrite, read_opts=ro, gate=gate,
                             geotiff=a.geotiff, **kw): p for p, kw in jobs}
        for fut in futures:
            name = futures[fut].name
            try:
                meta = fut.result()
            except Exception as e:   # keep going; report at the end
                failures += 1
                print(f"  {name}: FAILED {e}", file=sys.stderr)
                continue
            if meta is None:
                print(f"  {name}: cached")
                continue
            q = meta["quality"]
            suspect += q["suspect"]
            print(f"  {name}: {meta['seconds']} s, agree {q.get('agree_frac', float('nan')):.0%}"
                  + (f"  SUSPECT: {'; '.join(q['reasons'])}" if q["suspect"] else ""))
    print(f"done: {len(jobs) - failures} scenes ({suspect} suspect, skipped by training), {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

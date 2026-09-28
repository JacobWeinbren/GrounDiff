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
    <out>/<scene>/<channel>.npy          [H, W], NaN = no data; heights float32,
                                         density / z_std / echoes float16, masks uint8
Channels: dsm_max, dsm_min, dsm_last, density, z_std, echoes, has_return,
in_survey, dtm_before (+ before_valid), sem_ground, sem_nonground, gt_dtm
(+ gt_valid), flat_water, with --after-dir also top_ground, with --osm also
bridge (cells within 10 m of an OpenStreetMap bridge).

--osm <osm.json> (from groundiff.data.select): the hydro-flattened water under
and next to bridges (20 m) is kept in the target instead of being masked, so
the model learns to remove the deck down to the water. --selection
<selection.json> writes each tile's stratum (quarry, urban, marsh, ...) into
meta.json, for per-stratum evaluation.

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


MASK_CHANNELS = {"has_return", "in_survey", "gt_valid", "before_valid", "sem_ground", "sem_nonground",
                 "flat_water", "top_ground", "bridge"}
HALF_CHANNELS = {"density", "z_std", "echoes"}      # inputs only; float16 keeps 3 significant digits
BRIDGE_M, BRIDGE_KEEP_M = 10.0, 20.0


def _save(out: Path, name: str, arr: np.ndarray):
    """Readers cast to float32 (dataset windows, infer), so the smaller types are transparent."""
    if (name in MASK_CHANNELS or name.startswith(("unchanged_", "label_above"))) and not np.isnan(arr).any():
        dt = np.uint8
    elif name in HALF_CHANNELS:
        dt = np.float16
        arr = np.clip(arr, -65000, 65000)
    else:
        dt = np.float32
    np.save(out / f"{name}.npy", np.ascontiguousarray(arr, dtype=dt))


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


def quality(arrs: dict, alpha: float = 0.2, ground_tin: np.ndarray | None = None) -> dict:
    """Does the target belong to this point cloud?

    agree_frac / median_dz: agreement within alpha where lasground_new says
    ground (or, points only, on open ground against the lowest return).
    exact_frac: share of open-ground cells (single returns, < 5 cm spread)
    where the target equals a TIN of the tile's own ground points within
    5 mm. The EA DTM is a TIN of the same survey's edited ground at cell
    centres, so a same-survey DTM matches almost exactly on open ground
    (0.4-1.0 in tests), a DTM from another survey does not (0.002-0.05),
    although both pass agree_frac. ground_tin: the reference for points-only
    scenes (TIN of the published ground class, used for this check only)."""
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
        # offset from the histogram peak: heavy one-sided edits (e.g. roofs kept as ground) move the
        # median but not the mode
        h, e = np.histogram(np.clip(d, -1, 1), bins=200, range=(-1, 1))
        q["mode_dz"] = float(0.5 * (e[h.argmax()] + e[h.argmax() + 1]))
        q["agree_frac"] = float((np.abs(d) < alpha).mean())
    tin = arrs.get("dtm_before", ground_tin)
    if tin is not None:
        open_ = (ok & np.isfinite(tin) & (np.nan_to_num(arrs["z_std"], nan=9.0) < 0.05)
                 & (np.nan_to_num(arrs.get("echoes", np.ones_like(tin)), nan=9.0) <= 1.05))
        if "sem_ground" in arrs:
            open_ &= np.nan_to_num(arrs["sem_ground"]) > 0.5
        q["n_open_cells"] = int(open_.sum())
        if open_.sum() >= 200:
            q["exact_frac"] = float((np.abs(arrs["gt_dtm"] - tin)[open_] <= 0.005).mean())
    return q


def is_suspect(q: dict, min_agree: float = 0.6, max_offset: float = 0.15, min_coverage: float = 0.2,
               min_exact: float = 0.15) -> list:
    """min_coverage is only a sanity check: on the coast the EA DTM leaves the
    sea blank while the points include beach and sea returns, and only
    covered cells are trained on anyway."""
    why = []
    if min_exact and q.get("exact_frac") is not None and q["exact_frac"] < min_exact:
        why.append(f"only {q['exact_frac']:.0%} of open-ground cells match the tile's ground within 5 mm: "
                   "the DTM probably comes from another survey (if every tile says this, the DTMs are built "
                   "differently than assumed: rerun with --min-exact 0 and send the summary)")
    if q.get("canopy_gap_m") is not None and q["canopy_gap_m"] < 1.0:
        why.append(f"target follows the tree/building tops (median {q['canopy_gap_m']:.2f} m below the highest "
                   "return where there are several echoes): a DSM, not a DTM?")
    if q.get("target_coverage", 0) < min_coverage:
        why.append(f"target covers {q.get('target_coverage', 0):.0%} of the survey area")
    fallback = q.get("exact_frac") is None or not min_exact     # no survey-match test: use the coarse ones
    if (fallback and "agree_frac" in q and q.get("n_ground_cells", 0) >= 2000
            and q["agree_frac"] < min_agree):
        why.append(f"only {q['agree_frac']:.0%} of ground cells within 0.2 m of the target")
    offset = q.get("mode_dz", q.get("median_dz"))
    if offset is not None and abs(offset) > max_offset:
        why.append(f"offset {offset:+.2f} m between the target and the points on open ground")
    return why


def process_scene(before: Path, out_root: Path, *, dtm_paths: list | None = None, after: Path | None = None,
                  gsd: float = 1.0, ground_classes=(2,), before_ground_classes=(2,), lasground: bool = True,
                  overwrite: bool = False, read_opts: dict | None = None, coverage_close_m: float = 30.0,
                  gate: dict | None = None, geotiff: bool = False, bridges: list | None = None) -> dict | None:
    """before: point tile classified by lasground_new (lasground=False: any
    point tile, no dtm_before/sem channels). Target from dtm_paths (rasters)
    or after (hand-edited tile, ground_classes). read_opts: drop_withheld /
    drop_overlap / drop_synthetic / drop_classes for read_points.
    bridges: OpenStreetMap bridge centrelines near the tile ([[x, y], ...] in BNG).
    Returns meta, or None when an up-to-date cache exists."""
    before = Path(before)
    read_opts = dict(read_opts or {})
    name = scene_name(before)
    out = Path(out_root) / name
    meta_path = out / "meta.json"
    key = {"schema": SCHEMA, "gsd": gsd, "before": _stamp(before), "after": _stamp(after),
           "dtm": sorted(_stamp(Path(p)) for p in (dtm_paths or [])), "ground_classes": list(ground_classes),
           "before_ground_classes": list(before_ground_classes), "lasground": lasground,
           "read_opts": read_opts, "coverage_close_m": coverage_close_m}
    if bridges is not None:
        import hashlib
        key["bridges"] = hashlib.md5(json.dumps(bridges, sort_keys=True).encode()).hexdigest()
    key = json.loads(json.dumps(key))
    if meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text())
        stored = dict(meta.get("cache_key") or {})
        stored.pop("gate", None)               # older runs kept the thresholds in the key; they are re-applied below
        if stored == key:
            q = meta.get("quality")
            if q is not None:                  # re-apply the current quality thresholds to the stored statistics
                why = is_suspect({k: v for k, v in q.items() if k not in ("suspect", "reasons")}, **(gate or {}))
                if bool(why) != q.get("suspect") or why != q.get("reasons"):
                    q.update({"suspect": bool(why), "reasons": why})
                    meta_path.write_text(json.dumps(meta, indent=1))
            if geotiff:                        # cached scene, GeoTIFFs asked for now: write them from the .npy
                from ..io_raster import write_geotiff
                g = meta["grid"]
                for f in sorted(out.glob("*.npy")):
                    if not f.with_suffix(".tif").exists():
                        write_geotiff(f.with_suffix(".tif"), np.load(f).astype(np.float32), g["xmin"], g["ymax"], g["gsd"],
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
    ground_tin = None
    if not lasground and dtm_paths:            # published ground class: for the quality check only, never an input
        from .rasterise import tin_dtm
        g2 = pts.cls == 2
        if g2.sum() >= 3:
            ground_tin, ok_tin = tin_dtm(grid, pts.x[g2], pts.y[g2], pts.z[g2], need=survey)
            ground_tin = np.where(ok_tin & survey, ground_tin, np.nan)
    meta = {"schema": SCHEMA, "scene": name, "gsd": gsd, "grid": grid.to_dict(),
            "crs_wkt": pts.crs_wkt or _default_crs(), "before_file": str(before), "n_points": len(pts),
            "class_hist_before": hist, "has_before": lasground, "lasground": lasground,
            "before_ground_classes": list(before_ground_classes), "read_opts": read_opts}
    keep = None
    if bridges is not None:
        from .rasterise import line_mask
        arrs["bridge"] = line_mask(grid, bridges, BRIDGE_M).astype(np.float32)
        keep = line_mask(grid, bridges, BRIDGE_KEEP_M)
        meta["bridge_cells"] = int(arrs["bridge"].sum())
    if dtm_paths:
        arrs.update(target_from_rasters(grid, [str(p) for p in dtm_paths], survey, keep=keep))
        meta.update({"target": "dtm_raster", "dtm_files": [str(p) for p in dtm_paths]})
    else:
        ap = read_points(after, drop_withheld=read_opts.get("drop_withheld", True))
        arrs.update(target_from_points(grid, ap, survey, ground_classes))
        meta.update({"target": "after_points", "after_file": str(after), "ground_classes": list(ground_classes),
                     "class_hist_after": class_histogram(ap.cls)})
        if len(ap) != len(pts):
            meta["warning"] = f"before/after point counts differ ({len(pts)} vs {len(ap)})"
    q = quality(arrs, ground_tin=ground_tin)
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


DATES_RE = re.compile(r"(?<!\d)(20\d{6})_(20\d{6})(?!\d)")
SURVEY_RE = re.compile(r"_P_(\d+)_")


def survey_dates(name: str) -> tuple | None:
    """(start, end) flight dates from EA file names: TL4378nw_P_12534_20220315_20220316.laz,
    DTM_F0224538_20220319_20220319.tif, DTM_TM0010_P_12506_20220127_20220127.tif."""
    m = DATES_RE.search(Path(name).name)
    return (m.group(1), m.group(2)) if m else None


def survey_id(name: str) -> str | None:
    m = SURVEY_RE.search(Path(name).name)
    return m.group(1) if m else None


def match_survey(point_name: str, rasters: list[dict]) -> tuple[list[dict], str]:
    """Keep the DTM rasters of the point tile's own survey: same P number when
    the raster names carry one (NLP), else the same flight dates (time-stamped
    tiles are named by date, DTM_F<id>_<start>_<end>), else overlapping dates.
    Returns (rasters, how) with how = survey / dates / date-overlap / footprint."""
    pid, pd = survey_id(point_name), survey_dates(point_name)
    if pid:
        same = [r for r in rasters if survey_id(r["path"]) == pid]
        if same:
            return same, "survey"
    if pd:
        same = [r for r in rasters if survey_dates(r["path"]) == pd]
        if same:
            return same, "dates"
        over = [r for r in rasters if (d := survey_dates(r["path"])) and d[0] <= pd[1] and d[1] >= pd[0]]
        if over:
            return over, "date-overlap"
    return rasters, "footprint"


def rasters_for(bounds, index: list[dict], year: str | None = None, point_name: str | None = None
                ) -> list[str]:
    """Rasters covering bounds, best first (earlier ones win where they overlap).
    With point_name, only the rasters of the same survey are kept when they
    can be identified (see match_survey)."""
    x0, y0, x1, y1 = bounds
    hits = [r for r in index if r["xmin"] < x1 and r["xmax"] > x0 and r["ymin"] < y1 and r["ymax"] > y0]
    if point_name:
        hits, _ = match_survey(point_name, hits)
    hits.sort(key=lambda r: _raster_rank(r["path"], year) + (r["res"], r["path"]))
    return [r["path"] for r in hits]


def lines_near(lines: list, bounds, pad: float) -> list:
    x0, y0, x1, y1 = bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad
    out = []
    for ln in lines:
        xs, ys = [q[0] for q in ln], [q[1] for q in ln]
        if max(xs) >= x0 and min(xs) <= x1 and max(ys) >= y0 and min(ys) <= y1:
            out.append(ln)
    return out


def tag_strata(root: Path, selection: Path) -> None:
    """Copy each tile's stratum and OSM counts from selection.json into its meta.json."""
    sel = {scene_name(Path(t["name"])): t for t in json.loads(Path(selection).read_text())["tiles"]}
    for m in Path(root).glob("*/meta.json"):
        t = sel.get(m.parent.name)
        if t is None:
            continue
        meta = json.loads(m.read_text())
        if meta.get("stratum") != t["stratum"] or meta.get("osm") != t.get("osm"):
            meta.update({"stratum": t["stratum"], "osm": t.get("osm", {})})
            m.write_text(json.dumps(meta, indent=1))


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
    ap.add_argument("--min-exact", type=float, default=0.15,
                    help="quality gate: min share of open-ground cells matching the ground TIN within 5 mm "
                         "(catches DTMs from another survey); 0 disables")
    ap.add_argument("--geotiff", action="store_true", help="also write every channel as GeoTIFF")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--osm", type=Path, help="osm.json from groundiff.data.select: bridge channel, and water "
                    "under bridges kept in the target")
    ap.add_argument("--selection", type=Path, help="selection.json from groundiff.data.select: strata into meta.json")
    a = ap.parse_args(argv)

    before = find_laz(a.before_dir or a.points_dir)
    lasground = a.before_dir is not None
    a.out.mkdir(parents=True, exist_ok=True)
    ro = {"drop_overlap": a.drop_overlap, "drop_synthetic": a.drop_synthetic, "drop_withheld": not a.keep_withheld}
    gate = {"min_agree": a.min_agree, "max_offset": a.max_offset, "min_exact": a.min_exact}
    jobs = []
    osm_lines = json.loads(a.osm.read_text()).get("bridge", []) if a.osm else None
    if a.dtm_dir:
        from .laz import header_info
        index = index_rasters(a.dtm_dir, a.out / "dtm_index.json")
        match_counts = {}
        print(f"{len(index)} DTM rasters under {a.dtm_dir}")
        for k, p in before.items():
            try:
                b = header_info(p)["bounds"]
            except Exception as e:
                print(f"  {p.name}: FAILED reading header {e!r}", file=sys.stderr)
                continue
            from .ea_dtm import survey_year
            x0, y0, x1, y1 = b
            cover = [r for r in index if r["xmin"] < x1 and r["xmax"] > x0 and r["ymin"] < y1 and r["ymax"] > y0]
            hits = rasters_for(b, index, survey_year(p.name), p.name)
            if hits:
                _, how = match_survey(p.name, cover)
                match_counts[how] = match_counts.get(how, 0) + 1
            if not hits:
                print(f"[warn] {k}: no DTM raster covers it; skipped")
                continue
            kw = {"dtm_paths": hits}
            if osm_lines is not None:
                kw["bridges"] = lines_near(osm_lines, b, 100.0)
            jobs.append((p, kw))
    else:
        after = find_laz(a.after_dir)
        missing = sorted(set(before) - set(after))
        if missing:
            print(f"[warn] {len(missing)} tiles have no after file and are skipped: {missing[:5]}...")
        jobs = [(p, {"after": after[k]}) for k, p in before.items() if k in after]
    if a.dtm_dir and match_counts:
        print("DTM paired with its point tile by: " + ", ".join(f"{k} {v}" for k, v in sorted(match_counts.items()))
              + " (footprint = no survey id/dates to match; the quality gate checks those)")
    print(f"{len(jobs)} scenes -> {a.out}")
    failures, suspect, exact = 0, 0, []
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
            cached = meta is None
            if cached:
                meta = json.loads((a.out / scene_name(futures[fut]) / "meta.json").read_text())
            q = meta["quality"]
            suspect += bool(q["suspect"])
            if q.get("exact_frac") is not None:
                exact.append(q["exact_frac"])
            ex = q.get("exact_frac")
            print(f"  {name}: " + ("cached" if cached else f"{meta['seconds']} s")
                  + (f", 5 mm match {ex:.0%}" if ex is not None else "")
                  + f", coverage {q.get('target_coverage', float('nan')):.0%}"
                  + (f"  SUSPECT: {'; '.join(q['reasons'])}" if q["suspect"] else ""))
    if a.selection:
        tag_strata(a.out, a.selection)
    print(f"done: {len(jobs) - failures} scenes ({suspect} suspect, skipped by training), {failures} failed")
    if exact:
        qs = np.quantile(exact, [0.1, 0.25, 0.5, 0.75, 0.9])
        print("open-ground cells matching the ground TIN within 5 mm (same survey expected >= 0.4, other survey "
              "<= 0.05): 10/25/50/75/90 % = " + " / ".join(f"{v:.2f}" for v in qs))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

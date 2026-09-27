"""Many point-cloud tiles -> one seamless set of rasters (DTM, edit probability,
predicted edit, uncertainty), LP360 overlays and an editing priority list.

    python -m groundiff.batch --onnx models/before_after.onnx \\
        --tiles "lasground/*.laz" --out results/area1 --workers 3

Inputs are the tiles as lasground_new wrote them (default settings: every
point 1 = non-ground or 2 = ground); the model predicts where editors will
change that result. Tiles may be LAS/LAZ/COPC from any producer (EA, LAStools,
LP360); no points are dropped by class.

How tiles are processed
  * Each tile's extent comes from its header, checked against its point count
    and its OS grid name (e.g. TL4378nw = 500 m quadrant); a header that
    cannot be right is replaced by the bounds of the points themselves.
  * Every tile (or 6 km block of a larger one) is processed with a buffer of
    neighbouring points (default: one network tile + 32 m) and cropped back.
    Network tiles sit on one lattice anchored to the British National Grid
    origin and each tile's sampling noise is seeded by its position, so
    neighbouring jobs compute identical values where they overlap: no seams.
    Re-running part of an area reproduces the same numbers away from the edge
    of the new selection (within one buffer of it, neighbouring points are
    missing unless those tiles are selected too).
  * Each file is read once, chunk by chunk, straight into the rasters
    (data.stream): memory is ~50 bytes per cell plus 12 per ground point
    (about 2 GB for a 2 km tile at 1 m), not per point read. The ground TIN
    is built in blocks that give exactly one TIN of all points
    (rasterise.tin_dtm_local). The next tile is read while the network works
    on the current one; the log gives an estimate per run.
  * A tile that fails (unreadable file, wrong classes) is reported in
    batch_summary.json and the run carries on.

Outputs in --out: <name>.tif mosaics (+ .qml styles, .tfw/.prj, overviews),
<name>_overlay.tif (RGBA) / _overlay_rgb.tif for LP360, priority.csv and
priority.geojson (+ priority.shp where GDAL's Python bindings exist, e.g. in
QGIS): blocks ranked by predicted edit volume, with coordinates;
batch_summary.json; and tiles/ with the per-tile pieces.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .data.laz import data_bounds, header_info
from .data.osgrid import parse_tile
from .data.preprocess import check_lasground_classes, scene_name
from .data.rasterise import Grid
from .data.stream import Accumulator, stream_file
from .io_raster import (build_overviews, build_vrt, crs_wkt_from_epsg, epsg_of, iter_rows, vrt_to_geotiff,
                        write_geotiff)
from .overlay import _write_sidecars, qml_style, render_rgba, write_rgba_geotiff
from .runtime import RuntimeSpec, lattice_anchor, predict_scene


class Cancelled(Exception):
    pass


@dataclass
class Job:
    name: str
    core: tuple            # xmin, ymin, xmax, ymax (snapped to the grid)
    buffered: tuple
    files: list
    tile: str = ""


def _snap(b, gsd):
    return (math.floor(b[0] / gsd + 1e-9) * gsd, math.floor(b[1] / gsd + 1e-9) * gsd,
            math.ceil(b[2] / gsd - 1e-9) * gsd, math.ceil(b[3] / gsd - 1e-9) * gsd)


def _intersects(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def expand_inputs(paths: list) -> list[Path]:
    """Expand wildcards (Windows shells do not) and folders; drop duplicates."""
    out, seen = [], set()
    for p in paths:
        p = str(p).strip()
        if not p:                              # a blank entry would mean "the current folder"
            continue
        # a real path is used as is (names may contain [ ] which glob would treat as a pattern)
        hits = [p] if Path(p).exists() else sorted(glob.glob(p))
        for h in hits:
            hp = Path(h)
            if hp.is_dir() and (hp.resolve() == Path(hp.resolve().anchor) or hp.resolve() == Path.home()):
                raise ValueError(f"{p}: refusing to search a whole disk or home folder for LAS/LAZ files; "
                                 "pick the files or the folder that holds them")
            files = (_find_las(hp) if hp.is_dir()
                     else ([hp] if hp.suffix.lower() in (".las", ".laz") else []))
            if not files:
                print(f"[warn] {p}: matches no .las/.laz file")
            for f in files:
                key = str(f.resolve())
                key = key.lower() if sys.platform.startswith("win") else key
                if key not in seen:
                    seen.add(key)
                    out.append(f)
    # QGIS writes <name>.copc.laz next to a tile it loads: the same points, so keep the original
    names = {str(f.with_name(f.name[:-len(f.suffix)])).lower() for f in out
             if not f.name.lower().endswith(".copc.laz")}
    return [f for f in out if not (f.name.lower().endswith(".copc.laz")
                                   and str(f.with_name(f.name[:-len(".copc.laz")])).lower() in names)]


def _find_las(folder: Path) -> list[Path]:
    """LAS/LAZ files under folder; unreadable or offline sub-folders (cloud drives) are skipped."""
    found = []
    for root, dirs, names in os.walk(folder, onerror=lambda e: None):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        found += [Path(root) / n for n in names if n.lower().endswith((".las", ".laz"))]
    return sorted(found)


def tile_extent(path: Path, max_extent_m: float = 5000.0, log: Callable = print) -> tuple:
    """Trusted extent of a tile: header bounds if plausible, else the points'
    own bounds; clipped to the OS quadrant named in the file name (drops
    lastile-style buffers) when most of the tile lies inside it."""
    h = header_info(path)
    b, n = h["bounds"], h["point_count"]
    w, ht = b[2] - b[0], b[3] - b[1]
    density = n / max(w * ht, 1e-9)
    if not (w > 0 and ht > 0 and w <= max_extent_m and ht <= max_extent_m and 0.01 <= density <= 500):
        log(f"[warn] {path.name}: header bounds {tuple(round(v, 1) for v in b)} with {n} points look wrong; "
            "scanning the points for their real extent")
        b = data_bounds(path)
        if b is None:
            raise ValueError("no points")
    t = parse_tile(path.name)
    if t and t["size_known"]:            # 500 m quadrants; 4-digit names can be 1 or 2 km
        e = t["extent"]
        inter = (max(b[0], e[0]), max(b[1], e[1]), min(b[2], e[2]), min(b[3], e[3]))
        if inter[2] > inter[0] and inter[3] > inter[1]:
            area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
            if (inter[2] - inter[0]) * (inter[3] - inter[1]) >= 0.5 * area_b:
                b = inter
    return b


def plan(tiles: list, gsd: float, buffer_m: float, max_block_m: float = 6000.0,
         log: Callable = print) -> tuple[list[Job], tuple, dict]:
    """One job per tile (tiles larger than max_block_m are split into blocks).
    Returns (jobs, union of cores, {file name: error} for unusable files)."""
    tiles = [Path(p) for p in tiles]
    names: dict = {}
    for p in tiles:
        names.setdefault(scene_name(p), []).append(p)
    dup = {k: v for k, v in names.items() if len(v) > 1}
    if dup:
        lines = "; ".join(f"{k}: {', '.join(str(q) for q in v)}" for k, v in sorted(dup.items()))
        raise ValueError(f"the same tile name appears more than once (select each tile once): {lines}")
    extents, problems, seen = {}, {}, {}
    for p in tiles:
        try:
            h = header_info(p)
            sig = (tuple(round(v, 2) for v in h["bounds"]), h["point_count"])
            if sig in seen:                     # e.g. a renamed copy picked up by 'Add Directory'
                log(f"[warn] {p.name} has the same extent and point count as {seen[sig].name}; "
                    "treated as a copy and skipped")
                continue
            seen[sig] = p
            extents[p] = tile_extent(p, log=log)
        except Exception as e:
            problems[p.name] = f"cannot read: {e}"
    jobs = []
    for p, ext in extents.items():
        core = _snap(ext, gsd)
        nx = max(1, math.ceil((core[2] - core[0]) / max_block_m - 1e-9))
        ny = max(1, math.ceil((core[3] - core[1]) / max_block_m - 1e-9))
        xs = [core[0] + round((core[2] - core[0]) * i / nx / gsd) * gsd for i in range(nx)] + [core[2]]
        ys = [core[1] + round((core[3] - core[1]) * j / ny / gsd) * gsd for j in range(ny)] + [core[3]]
        for i in range(nx):
            for j in range(ny):
                blk = (xs[i], ys[j], xs[i + 1], ys[j + 1])
                buf = _snap((blk[0] - buffer_m, blk[1] - buffer_m, blk[2] + buffer_m, blk[3] + buffer_m), gsd)
                src = [q for q, e in extents.items() if _intersects(e, buf)]
                name = scene_name(p) + (f"_b{i:02d}_{j:02d}" if nx * ny > 1 else "")
                jobs.append(Job(name, blk, buf, src, p.name))
    names = [j.name for j in jobs]
    assert len(set(names)) == len(names), "duplicate job names"
    if not jobs:
        return [], (0.0, 0.0, 0.0, 0.0), problems
    cores = [j.core for j in jobs]
    union = (min(c[0] for c in cores), min(c[1] for c in cores),
             max(c[2] for c in cores), max(c[3] for c in cores))
    return jobs, union, problems


def prepare_job(job: Job, gsd: float, read_opts: dict | None = None, lasground: bool | str = True,
                before_ground_classes=(2,), progress: Callable | None = None,
                cancelled: Callable = lambda: False) -> tuple[dict, Grid, dict]:
    """lasground: True = the model needs lasground_new classes (error if the
    tiles do not look like lasground_new output); "optional" = build the
    ground rasters if the tiles have class 2 (class 2 = ground, everything
    else not: EA production also keeps bridge / noise classes), for
    dz_before / p_edit of models that do not use them as inputs; False =
    never.
    Each file is read once, chunk by chunk, straight into the rasters
    (data.stream); progress(fraction) follows the points read."""
    t0 = time.time()
    grid = Grid(job.buffered[0], job.buffered[3], gsd, int(round((job.buffered[2] - job.buffered[0]) / gsd)),
                int(round((job.buffered[3] - job.buffered[1]) / gsd)))
    acc = Accumulator(grid, keep_ground=bool(lasground), ground_classes=before_ground_classes)
    sizes = []
    for f in job.files:
        try:
            sizes.append(max(header_info(f)["point_count"], 1))
        except Exception:
            sizes.append(1)
    done = 0.0
    for f, n in zip(job.files, sizes):
        def prog(fr, _done=done, _n=n):
            if progress:
                progress((_done + fr * _n) / sum(sizes))
        try:
            stream_file(f, job.buffered, acc, read_opts, progress=prog, cancelled=cancelled)
        except RuntimeError as e:
            if str(e) == "cancelled":
                raise Cancelled() from e
            raise RuntimeError(f"cannot read {Path(f).name}: {e}") from e
        done += n
    info = {"n_points": acc.n_points, "crs_wkt": acc.crs_wkt}
    if acc.n_points == 0:
        return {}, grid, info
    use_classes = False
    if lasground:
        hist = acc.class_histogram()
        info["classes"] = hist
        if lasground is True:                  # the model takes lasground_new's classes as an input
            problem = check_lasground_classes(hist)
            if problem:
                raise ValueError(f"{problem} (classes {hist}); this model needs tiles classified by lasground_new")
            use_classes = True
        else:                                  # only the reference DTM: class 2 is ground, all else is not
            use_classes = bool(hist.get(2))
        info["lasground_classes"] = use_classes
    if progress:
        progress(1.0)
    info["read_s"] = round(time.time() - t0, 1)
    t1 = time.time()
    arrs = acc.rasters(use_classes)
    info["tin_s"] = round(time.time() - t1, 1)
    return arrs, grid, info


def output_keys(spec: RuntimeSpec, predict_kwargs: dict | None = None) -> list[str]:
    """Rasters this model produces (so callers only ask for those)."""
    predict_kwargs = predict_kwargs or {}
    keys = ["dtm"]
    edit = spec.kind == "groundiff" and bool(spec.prior_channel) and spec.gate_channel == spec.prior_channel
    if spec.kind == "groundiff":
        # DSM -> DTM models: p_edit (share of samples differing from lasground_new by > alpha) and
        # dz_before exist when the input tiles carry lasground_new classes
        keys += ["p_edit"] if edit else ["p_ground", "p_edit"]
    if spec.kind == "groundiff" or spec.prior_channel:
        keys.append("dz_before")
    if predict_kwargs.get("n_samples", 1) > 1 or predict_kwargs.get("tta"):
        keys.append("std")
    return keys


OUTPUT_PRESETS = {"p_edit": "edit", "dz_before": "dz", "std": "uncertainty"}

# outline-only squares (so the edit rasters stay visible), labelled with their rank for the top 50
PRIORITY_QML = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22" styleCategories="Symbology|Labeling">
 <renderer-v2 type="singleSymbol">
  <symbols>
   <symbol type="fill" name="0" alpha="1">
    <layer class="SimpleFill">
     <Option type="Map">
      <Option name="style" value="no" type="QString"/>
      <Option name="outline_color" value="58,6,80,255" type="QString"/>
      <Option name="outline_width" value="0.5" type="QString"/>
      <Option name="outline_width_unit" value="MM" type="QString"/>
     </Option>
     <prop k="style" v="no"/>
     <prop k="outline_color" v="58,6,80,255"/>
     <prop k="outline_width" v="0.5"/>
     <prop k="outline_width_unit" v="MM"/>
    </layer>
   </symbol>
  </symbols>
 </renderer-v2>
 <labeling type="simple">
  <settings calloutType="simple">
   <text-style fieldName="CASE WHEN &quot;rank&quot; &lt;= 50 THEN &quot;rank&quot; END" isExpression="1"
               fontSize="9" textColor="58,6,80,255"/>
   <text-buffer bufferDraw="1" bufferSize="0.8" bufferColor="255,255,255,255"/>
   <placement placement="1"/>
  </settings>
 </labeling>
</qgis>
"""


class Priorities:
    """Per-block statistics on a fixed block lattice (absolute coordinates,
    so blocks never depend on how the area was split into jobs)."""

    def __init__(self, block_m: float, alpha: float):
        self.block_m, self.alpha = float(block_m), float(alpha)
        self.acc: dict = {}

    def add(self, x0: float, y1: float, gsd: float, dz: np.ndarray | None, p_edit: np.ndarray | None,
            std: np.ndarray | None):
        ref = dz if dz is not None else p_edit
        if ref is None:
            return
        H, W = ref.shape
        xs = x0 + (np.arange(W) + 0.5) * gsd
        ys = y1 - (np.arange(H) + 0.5) * gsd
        BX, BY = np.meshgrid(np.floor(xs / self.block_m).astype(np.int64),
                             np.floor(ys / self.block_m).astype(np.int64))
        ok = np.isfinite(ref)
        if not ok.any():
            return
        uniq, inv = np.unique(np.stack([BX[ok], BY[ok]], 1), axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        m = len(uniq)
        cols = {"n": np.bincount(inv, minlength=m).astype(float)}
        if dz is not None:
            ad = np.abs(np.nan_to_num(dz[ok]))
            cols["sum_dz"] = np.bincount(inv, ad, m)
            cols["n_edit"] = np.bincount(inv, (ad > self.alpha).astype(float), m)
            mx = np.zeros(m)
            np.maximum.at(mx, inv, ad)
            cols["max_dz"] = mx
        if p_edit is not None:
            pe = np.nan_to_num(p_edit[ok])
            cols["sum_p"] = np.bincount(inv, pe, m)
            cols["n_p"] = np.bincount(inv, (pe > 0.5).astype(float), m)
        if std is not None:
            cols["sum_std"] = np.bincount(inv, np.nan_to_num(std[ok]), m)
        for i, (kx, ky) in enumerate(uniq):
            a = self.acc.setdefault((int(kx), int(ky)), {})
            for c, v in cols.items():
                a[c] = max(a.get(c, 0.0), float(v[i])) if c == "max_dz" else a.get(c, 0.0) + float(v[i])

    def add_mosaics(self, outputs: dict, G: Grid, rows: int = 1024):
        """Accumulate from the finished mosaics (one value per cell, so tiles
        whose extents overlap are never counted twice)."""
        names = [k for k in ("dz_before", "p_edit", "std") if k in outputs]
        if not ({"dz_before", "p_edit"} & set(names)):
            return
        its = {k: iter_rows(outputs[k], rows) for k in names}
        for r0 in range(0, G.height, rows):
            blocks = {k: next(its[k])[1] for k in names}
            self.add(G.xmin, G.ymax - r0 * G.gsd, G.gsd, blocks.get("dz_before"), blocks.get("p_edit"),
                     blocks.get("std"))

    def rows(self, gsd: float) -> list[dict]:
        out = []
        for (kx, ky), a in self.acc.items():
            n = max(a["n"], 1.0)
            r = {"x_min": kx * self.block_m, "y_min": ky * self.block_m,
                 "x_centre": (kx + 0.5) * self.block_m, "y_centre": (ky + 0.5) * self.block_m,
                 "area_m2": a["n"] * gsd * gsd}
            if "sum_dz" in a:
                r.update({"mean_abs_dz_m": a["sum_dz"] / n, "max_abs_dz_m": a["max_dz"],
                          "edit_area_m2": a["n_edit"] * gsd * gsd, "edit_volume_m3": a["sum_dz"] * gsd * gsd})
            if "sum_p" in a:
                r.update({"mean_p_edit": a["sum_p"] / n, "p_edit_area_m2": a["n_p"] * gsd * gsd})
            if "sum_std" in a:
                r["mean_std_m"] = a["sum_std"] / n
            out.append(r)
        key = "edit_volume_m3" if out and "edit_volume_m3" in out[0] else "p_edit_area_m2"
        out.sort(key=lambda r: -r.get(key, 0.0))
        for i, r in enumerate(out, 1):
            r["rank"] = i
        return out

    def write(self, out_dir: Path, gsd: float, epsg: int | None) -> list[str]:
        rows = self.rows(gsd)
        if not rows:
            return []
        fields = ["rank"] + [k for k in rows[0] if k != "rank"]
        rnd = lambda r: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
        csv_path = out_dir / "priority.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rnd(r) for r in rows)
        b = self.block_m
        feats = [{"type": "Feature", "properties": rnd(r),
                  "geometry": {"type": "Polygon", "coordinates": [[
                      [r["x_min"], r["y_min"]], [r["x_min"] + b, r["y_min"]], [r["x_min"] + b, r["y_min"] + b],
                      [r["x_min"], r["y_min"] + b], [r["x_min"], r["y_min"]]]]}} for r in rows]
        gj = {"type": "FeatureCollection", "features": feats}
        if epsg:
            gj["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}}
        gj_path = out_dir / "priority.geojson"
        gj_path.write_text(json.dumps(gj))
        (out_dir / "priority.qml").write_text(PRIORITY_QML)
        written = [str(csv_path), str(gj_path)]
        try:                            # shapefile for LP360 when GDAL's Python bindings exist (QGIS)
            from osgeo import gdal
            shp = out_dir / "priority.shp"
            if gdal.VectorTranslate(str(shp), str(gj_path), format="ESRI Shapefile") is not None:
                written.append(str(shp))
        except Exception:
            pass
        return written


def _job_points(job: Job, counts: dict, extents: dict) -> float:
    """Rough number of points a job reads (header counts x overlap share)."""
    n = 0.0
    b = job.buffered
    for f in job.files:
        e = extents.get(f)
        if e is None:
            continue
        ix = max(0.0, min(b[2], e[2]) - max(b[0], e[0]))
        iy = max(0.0, min(b[3], e[3]) - max(b[1], e[1]))
        area = max((e[2] - e[0]) * (e[3] - e[1]), 1e-9)
        n += counts.get(f, 0) * min(1.0, ix * iy / area)
    return n


def run_batch(tiles: list, out_dir: str | Path, net, spec: RuntimeSpec, *, gsd: float | None = None,
              buffer_m: float | None = None, workers: int = 1, read_opts: dict | None = None,
              overlays: bool = True, predict_kwargs: dict | None = None, max_block_m: float = 6000.0,
              block_m: float = 100.0, progress: Callable | None = None, log: Callable = print,
              cancelled: Callable = lambda: False, default_epsg: int | None = 27700,
              status: Callable | None = None) -> dict:
    """tiles: LAS/LAZ/COPC paths, folders or wildcards (lasground_new output
    for models that use its classes). gsd / read_opts default to the values
    the model was trained with (spec). default_epsg: CRS for outputs when
    the inputs carry none (EA open-data tiles have no CRS VLR; they are
    EPSG:27700). Raises Cancelled if cancelled() becomes true.
    progress(fraction): tiles fill 0-95 %, mosaics and priorities the rest.
    status(text): what is happening now, e.g. 'Tile 3/12 SP1234: model 40 %
    - about 14 min left' (QGIS: feedback.setProgressText)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gsd = float(gsd or spec.gsd or 1.0)
    if spec.gsd and abs(gsd - spec.gsd) > 1e-9:
        log(f"[warn] the model was trained at {spec.gsd} m cells; running at {gsd} m")
    if not spec.gsd:
        log(f"[warn] the model file does not record its training cell size; using {gsd} m (re-export the model "
            "with this version to record it)")
    need = (spec.tile - 1) * gsd
    buffer_m = float(buffer_m) if buffer_m is not None else spec.tile * gsd + 32.0
    if buffer_m < need:
        log(f"[warn] buffer {buffer_m:g} m is smaller than one network tile ({need:g} m): values may differ "
            "slightly where tiles meet")
    ro = dict(spec.read_opts or {})
    ro.update(read_opts or {})
    lasground = True if spec.needs_before else "optional"
    predict_kwargs = dict(predict_kwargs or {})
    seed = int(predict_kwargs.pop("seed", 0))
    files = expand_inputs(tiles)
    if not files:
        raise ValueError("no input tiles")
    for f in out.glob("priority.*"):           # never leave (and load) an earlier run's priority list
        f.unlink()
    jobs, union, problems = plan(files, gsd, buffer_m, max_block_m, log)
    for name, err in problems.items():
        log(f"[error] {name}: {err}")
    if not jobs:
        raise ValueError("none of the input tiles could be read: "
                         + "; ".join(f"{k}: {v}" for k, v in problems.items()))
    G = Grid.from_bounds(*union, gsd)
    counts, extents = {}, {}
    try:                                       # memory: ~50 bytes per cell + 12 per ground point + TIN blocks
        infos = {Path(f): header_info(f) for f in {f for j in jobs for f in j.files}}
        counts = {f: i["point_count"] for f, i in infos.items()}
        extents = {f: i["bounds"] for f, i in infos.items()}
        worst = max(jobs, key=lambda j: _job_points(j, counts, extents))
        pts = _job_points(worst, counts, extents)
        cells = (worst.buffered[2] - worst.buffered[0]) * (worst.buffered[3] - worst.buffered[1]) / gsd ** 2
        gb = (cells * 60 + pts * 0.6 * 12) / 1e9 + 1.0
        log(f"[info] largest tile: ~{pts / 1e6:.0f}M points; reading it needs about {gb:.1f} GB RAM")
    except Exception:
        pass
    keys = output_keys(spec, predict_kwargs)
    ov_keys = [k for k in keys if k in OUTPUT_PRESETS] if overlays else []
    tiles_dir = out / "tiles"
    tiles_dir.mkdir(exist_ok=True)
    placed = {k: [] for k in keys}
    placed.update({f"{k}{sfx}": [] for k in ov_keys for sfx in ("_overlay", "_overlay_rgb")})
    prio = Priorities(block_m, spec.alpha)
    summary = {"tiles": [], "failed": [{"file": k, "error": v} for k, v in problems.items()],
               "grid": G.to_dict(), "gsd": gsd, "buffer_m": buffer_m, "n_jobs": len(jobs), "cancelled": False}
    crs, crs_is_default = None, False
    t_start = time.time()
    TILES_SHARE = 0.95
    READ_SHARE = 0.3                           # of each tile's slot (reads overlap the previous tile's model run)
    done_frac = [0.0]

    def say(text):
        if status:
            status(text)

    def report(f):
        """overall fraction f of the tile work done"""
        done_frac[0] = f
        if progress:
            progress(TILES_SHARE * f)

    tile_secs = []                             # wall time of each finished tile
    cur = {"start": time.time(), "model_start": None, "f": 0.0}

    def left():
        """time left from measured rates: the running model's own speed, and finished tiles for the rest"""
        now = time.time()
        idx_done = len(tile_secs)
        rest = len(jobs) - idx_done - 1
        per_tile = sum(tile_secs) / idx_done if idx_done else None
        f, ms = cur["f"], cur["model_start"]
        if ms is not None and f >= 0.02 and now - ms >= 10:
            model_total = (now - ms) / f
            this = model_total * (1 - f)
            if per_tile is None:
                per_tile = (ms - cur["start"]) + model_total
        elif per_tile is not None:
            this = max(per_tile - (now - cur["start"]), 0.0)
        else:
            return ""
        return eta_seconds(this + rest * per_tile)

    def check_cancel(_frac=None):
        if cancelled():
            raise Cancelled()

    ex = ThreadPoolExecutor(max_workers=max(1, workers))
    pending, nxt, read_frac = {}, 0, {}
    warned_classes = False

    def submit_upto(limit):
        nonlocal nxt
        while nxt < len(jobs) and len(pending) < limit:
            read_frac[nxt] = 0.0
            pending[nxt] = ex.submit(prepare_job, jobs[nxt], gsd, ro, lasground,
                                     tuple(spec.before_ground_classes or (2,)),
                                     lambda f, _k=nxt: read_frac.__setitem__(_k, f), cancelled)
            nxt += 1

    try:
        # workers=1: the next tile is read while this one runs through the model (2 tiles in memory)
        submit_upto(max(1, workers))
        for idx, job in enumerate(jobs):
            check_cancel()
            rec = {"name": job.name, "tile": job.tile}
            fut = pending.pop(idx)
            label = f"Tile {idx + 1}/{len(jobs)} {job.name}"
            t_read = time.time()
            while not fut.done():                  # stay responsive to cancel during a slow read
                check_cancel()
                fr = read_frac.get(idx, 0.0)
                report((idx + READ_SHARE * fr) / len(jobs))
                say(f"{label}: " + (f"reading points {100 * fr:.0f} %" if fr < 1 else "making the rasters")
                    + f", {time.time() - t_read:.0f} s{left()}")
                wait([fut], timeout=0.5)
            try:
                arrs, grid, info = fut.result()
            except (ImportError, Cancelled):
                raise                            # missing package: stop, the caller shows how to install it
            except Exception as e:
                msg = str(e)
                if "LazBackend" in msg or "lazrs" in msg or "laszip" in msg:
                    raise ImportError(f"LAZ support missing ({msg}); install laspy[lazrs]") from e
                arrs, rec["error"] = None, msg
            submit_upto(max(1, workers))
            if arrs is None:
                log(f"[error] {job.name}: {rec['error']}")
                summary["failed"].append({"file": job.tile, "job": job.name, "error": rec["error"]})
                continue
            rec.update({k: v for k, v in info.items() if k != "crs_wkt"})
            log(f"{job.name}: {info['n_points'] / 1e6:.1f}M points read in {info.get('read_s')} s, rasters "
                f"{'with' if info.get('lasground_classes') else 'without'} the ground TIN in "
                f"{info.get('tin_s')} s; classes {info.get('classes')}")
            if lasground == "optional" and info.get("lasground_classes") is False and not warned_classes:
                warned_classes = True
                log(f"[info] {job.name} has no ground (class 2) points, so no ground DTM, predicted edit or "
                    "edit probability can be made for it: only the predicted DTM.")
            if not arrs:
                rec["skipped"] = "no points"
                summary["tiles"].append(rec)
                continue
            if crs is None:
                crs = info["crs_wkt"]
                if crs is None and default_epsg:
                    crs = crs_wkt_from_epsg(default_epsg)
                    crs_is_default = True
                    if crs:
                        log(f"[info] inputs have no CRS; writing outputs as EPSG:{default_epsg}")
                    else:
                        log(f"[warn] inputs have no CRS and EPSG:{default_epsg} could not be built; "
                            "outputs have no CRS")
            # the job's core in its buffered grid, and its place on the mosaic
            c0 = int(round((job.core[0] - grid.xmin) / gsd))
            r0 = int(round((grid.ymax - job.core[3]) / gsd))
            w = int(round((job.core[2] - job.core[0]) / gsd))
            h = int(round((job.core[3] - job.core[1]) / gsd))
            anchor = lattice_anchor(grid.xmin, grid.ymax, gsd)
            t1 = time.time()
            cur["model_start"], cur["f"] = t1, 0.0
            say(f"{label}: model starting ({predict_kwargs.get('n_samples', 1)} sample(s))")

            def prog(f, _idx=idx, _label=label):
                check_cancel()
                cur["f"] = f
                report((_idx + READ_SHARE + (1 - READ_SHARE) * f) / len(jobs))
                say(f"{_label}: model {100 * f:.0f} %{left()}")

            try:
                res = predict_scene(arrs, spec, net, seed=seed, anchor=anchor, region=(r0, c0, h, w), gsd=gsd,
                                    progress=prog, **predict_kwargs)
            except Cancelled:
                raise
            except Exception as e:
                rec["error"] = f"prediction failed: {e!r}"
                log(f"[error] {job.name}: {rec['error']}")
                summary["failed"].append({"file": job.tile, "job": job.name, "error": rec["error"]})
                continue
            rec["predict_s"] = round(time.time() - t1, 1)
            tile_secs.append(time.time() - cur["start"])
            cur.update(start=time.time(), model_start=None, f=0.0)
            gc = int(round((job.core[0] - G.xmin) / gsd))
            gr = int(round((G.ymax - job.core[3]) / gsd))
            geo = (job.core[0], job.core[3], gsd, crs)
            crop = {k: res[k][r0:r0 + h, c0:c0 + w] for k in keys if k in res}
            for k, a in crop.items():
                p = tiles_dir / f"{job.name}_{k}.tif"
                write_geotiff(p, a, *geo)
                placed[k].append((p, gr, gc, h, w))
            for k in ov_keys:
                if k in crop:
                    rgba = render_rgba(crop[k], OUTPUT_PRESETS[k],
                                       crop.get("dz_before") if k == "p_edit" else None)
                    for sfx, alpha in (("_overlay", True), ("_overlay_rgb", False)):
                        p = tiles_dir / f"{job.name}_{k}{sfx}.tif"
                        write_rgba_geotiff(p, rgba, *geo, alpha=alpha)
                        placed[f"{k}{sfx}"].append((p, gr, gc, h, w))
            summary["tiles"].append(rec)
            log(f"{job.name}: {info['n_points']} pts, read {info.get('read_s')} s, predict {rec['predict_s']} s")
            report((idx + 1) / len(jobs))
    except BaseException as e:            # cancel, Ctrl-C or an unexpected error: record and stop cleanly
        summary["cancelled"] = isinstance(e, (Cancelled, KeyboardInterrupt))
        if not summary["cancelled"]:
            summary["error"] = repr(e)
        summary["seconds"] = round(time.time() - t_start, 1)
        (out / "batch_summary.json").write_text(json.dumps(summary, indent=1))
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    if lasground == "optional" and not any(t.get("lasground_classes") for t in summary["tiles"]):
        log("[info] the tiles have no ground (class 2) points, so there is no predicted edit "
            "(dz_before / p_edit), only the predicted DTM")
    if not any("skipped" not in t for t in summary["tiles"]):
        (out / "batch_summary.json").write_text(json.dumps(summary, indent=1))
        errs = "; ".join(f"{f.get('job', f['file'])}: {f['error']}" for f in summary["failed"][:5])
        raise ValueError(f"no tile could be processed. {errs}")

    # mosaics: VRT index over the tile files, then one compressed GeoTIFF each
    outputs = {}
    n_mos = sum(1 for v in placed.values() if v)

    def mosaic_step(i, name):
        check_cancel()
        say(f"Joining tiles into one raster: {name} ({i + 1}/{n_mos})")
        if progress:
            progress(TILES_SHARE + (1 - TILES_SHARE) * 0.9 * i / max(1, n_mos))
    try:
        _mosaic(placed, outputs, out, G, gsd, crs, step=mosaic_step)
    except BaseException as e:
        summary["error"] = f"writing mosaics failed: {e!r}"
        summary["outputs"] = outputs
        (out / "batch_summary.json").write_text(json.dumps(summary, indent=1))
        raise
    summary["outputs"] = outputs
    say("Ranking priority blocks")
    if progress:
        progress(TILES_SHARE + (1 - TILES_SHARE) * 0.9)
    prio.add_mosaics(outputs, G)
    epsg = default_epsg if crs_is_default else epsg_of(crs)
    summary["priority"] = prio.write(out, gsd, epsg)
    summary["seconds"] = round(time.time() - t_start, 1)
    if summary["failed"]:
        log(f"[warn] {len(summary['failed'])} tiles/blocks failed; see batch_summary.json")
    (out / "batch_summary.json").write_text(json.dumps(summary, indent=1))
    if progress:
        progress(1.0)
    say(f"Done in {summary['seconds'] / 60:.0f} min" if summary["seconds"] >= 60 else f"Done in {summary['seconds']:.0f} s")
    return summary


def eta_seconds(rem: float) -> str:
    if rem < 90:
        return " - under 2 min left"
    if rem < 5400:
        return f" - about {rem / 60:.0f} min left"
    return f" - about {rem / 3600:.1f} h left"


def eta_text(elapsed: float, frac: float) -> str:
    """' - about 14 min left' once there is enough progress to guess."""
    if frac < 0.02 or elapsed < 20:
        return ""
    rem = elapsed * (1 - frac) / frac
    if rem < 90:
        return " - under 2 min left"
    if rem < 5400:
        return f" - about {rem / 60:.0f} min left"
    return f" - about {rem / 3600:.1f} h left"


def _mosaic(placed: dict, outputs: dict, out: Path, G: Grid, gsd: float, crs, step: Callable | None = None):
    for i, (name, items) in enumerate((n, v) for n, v in placed.items() if v):
        if step:
            step(i, name)
        rgb = name.endswith("_overlay") or name.endswith("_overlay_rgb")
        tif = out / f"{name}.tif"
        if rgb:
            rgba = name.endswith("_overlay")
            vrt = build_vrt(out / f"{name}.vrt", items, G.width, G.height, G.xmin, G.ymax, gsd, crs,
                            bands=4 if rgba else 3, dtype="Byte", nodata=None if rgba else 0,
                            colorinterp=["Red", "Green", "Blue", "Alpha"] if rgba else ["Red", "Green", "Blue"])
            vrt_to_geotiff(vrt, tif, rgba=rgba)
            build_overviews(tif, "nearest")
        else:
            vrt = build_vrt(out / f"{name}.vrt", items, G.width, G.height, G.xmin, G.ymax, gsd, crs)
            vrt_to_geotiff(vrt, tif)
            build_overviews(tif, "average")
            if name in OUTPUT_PRESETS:
                tif.with_name(tif.stem + ".qml").write_text(qml_style(OUTPUT_PRESETS[name]))
        _write_sidecars(tif, G.xmin, G.ymax, gsd, crs)
        outputs[name] = str(tif)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint")
    src.add_argument("--onnx")
    ap.add_argument("--tiles", nargs="+", required=True,
                    help="lasground_new-classified LAS/LAZ/COPC files, folders or wildcards (quote them)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gsd", type=float, help="cell size (default: as trained)")
    ap.add_argument("--buffer", type=float, help="metres of neighbouring points (default: one network tile + 32 m)")
    ap.add_argument("--workers", type=int, default=1,
                    help="tiles read/rasterised in the background (each needs a few GB RAM; 1 already overlaps "
                         "reading with prediction)")
    ap.add_argument("--device", default="auto",
                    help="auto | cuda | coreml | directml | cpu with --onnx (auto: NVIDIA GPU, else the speed test's "
                         "choice, else CPU); auto | cuda | mps | cpu with --checkpoint")
    ap.add_argument("--blend", choices=["min", "linear", "mean"], default="linear")
    ap.add_argument("--prior", choices=["auto", "global", "channel", "none"], default="auto")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-m", type=float, default=100.0, help="priority block size (m)")
    ap.add_argument("--drop-overlap", action="store_true")
    ap.add_argument("--drop-synthetic", action="store_true")
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("--epsg", type=int, default=27700, help="CRS for outputs when inputs have none")
    a = ap.parse_args(argv)
    from .infer import load_net
    net, spec = load_net(a.checkpoint, a.onnx, a.device)
    ro = {}
    if a.drop_overlap:
        ro["drop_overlap"] = True
    if a.drop_synthetic:
        ro["drop_synthetic"] = True
    s = run_batch(a.tiles, a.out, net, spec, gsd=a.gsd, buffer_m=a.buffer, workers=a.workers,
                  overlays=not a.no_overlays, read_opts=ro, default_epsg=a.epsg, block_m=a.block_m,
                  predict_kwargs={"blend": a.blend, "prior": a.prior, "n_samples": a.samples, "tta": a.tta,
                                  "batch_size": a.batch_size})
    print(json.dumps({k: v for k, v in s.items() if k != "tiles"}, indent=1))
    return 1 if s["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

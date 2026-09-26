"""Many point-cloud tiles -> one seamless set of rasters (DTM, edit probability,
predicted edit, uncertainty and LP360 overlays).

    python -m groundiff.batch --onnx models/before_after.onnx \
        --after EA/*.laz --before LASGROUND/*.laz --out results/area1 --gsd 0.5 --workers 3

Each tile is processed with a buffer of points from its neighbours (default
64 m) and then cropped back to its own extent, so there are no seams where
tiles meet. Tiles are read and rasterised in background threads while the
network works on the previous one; results are written straight into the
mosaic GeoTIFFs, so memory stays bounded by one buffered tile. "Before" files
are matched to "after" files by file name (".copc" ignored).

Also used by the QGIS plugin (no torch needed with --onnx).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .data.laz import NOISE_CLASSES, concat, header_bounds, read_points_bbox
from .data.preprocess import scene_name
from .data.rasterise import Grid, build_rasters
from .io_raster import build_vrt, vrt_to_geotiff, write_geotiff
from .overlay import PRESETS, _write_sidecars, qml_style, render_rgba, write_rgba_geotiff
from .runtime import RuntimeSpec, predict_scene


@dataclass
class Job:
    name: str
    core: tuple            # xmin, ymin, xmax, ymax (snapped to the grid)
    buffered: tuple
    after_files: list
    before_files: list
    missing_before: list = field(default_factory=list)


def _snap(b, gsd):
    return (math.floor(b[0] / gsd) * gsd, math.floor(b[1] / gsd) * gsd,
            math.ceil(b[2] / gsd) * gsd, math.ceil(b[3] / gsd) * gsd)


def _intersects(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def plan(after_files: list, before_files: list | None, gsd: float, buffer_m: float = 64.0,
         max_block_m: float = 2000.0) -> tuple[list[Job], tuple]:
    """One job per tile, split into blocks of at most max_block_m on a side
    (keeps memory bounded for large tiles)."""
    after_files = [Path(p) for p in after_files]
    bounds = {p: header_bounds(p) for p in after_files}
    before_map = {scene_name(Path(p)): Path(p) for p in (before_files or [])}
    jobs = []
    for p in after_files:
        core = _snap(bounds[p], gsd)
        nx = max(1, math.ceil((core[2] - core[0]) / max_block_m))
        ny = max(1, math.ceil((core[3] - core[1]) / max_block_m))
        xs = [core[0] + round((core[2] - core[0]) * i / nx / gsd) * gsd for i in range(nx)] + [core[2]]
        ys = [core[1] + round((core[3] - core[1]) * j / ny / gsd) * gsd for j in range(ny)] + [core[3]]
        for i in range(nx):
            for j in range(ny):
                blk = (xs[i], ys[j], xs[i + 1], ys[j + 1])
                buf = (blk[0] - buffer_m, blk[1] - buffer_m, blk[2] + buffer_m, blk[3] + buffer_m)
                src = [q for q in after_files if _intersects(bounds[q], buf)]
                name = scene_name(p) + (f"_b{i}{j}" if nx * ny > 1 else "")
                job = Job(name, blk, buf, src, [])
                if before_files:
                    for q in src:
                        k = scene_name(q)
                        (job.before_files.append(before_map[k]) if k in before_map
                         else job.missing_before.append(k))
                jobs.append(job)
    cores = [j.core for j in jobs]
    union = (min(c[0] for c in cores), min(c[1] for c in cores),
             max(c[2] for c in cores), max(c[3] for c in cores))
    return jobs, union


def prepare_job(job: Job, gsd: float, read_opts: dict | None = None, ground_classes=(2, 9),
                before_ground_classes=(2,)) -> tuple[dict, Grid, dict]:
    read_opts = read_opts or {}
    t0 = time.time()
    after = concat([read_points_bbox(f, job.buffered, drop_classes=NOISE_CLASSES, **read_opts)
                    for f in job.after_files])
    before = (concat([read_points_bbox(f, job.buffered, drop_classes=NOISE_CLASSES, **read_opts)
                      for f in job.before_files]) if job.before_files else None)
    grid = Grid.from_bounds(*job.buffered, gsd)
    info = {"n_points": len(after), "n_points_before": len(before) if before is not None else 0,
            "crs_wkt": after.crs_wkt}
    if len(after) == 0:
        return {}, grid, info
    arrs = build_rasters(grid, after, before, ground_classes, before_ground_classes, with_target=False)
    info["read_s"] = round(time.time() - t0, 1)
    return {k: v.astype(np.float64) for k, v in arrs.items()}, grid, info


OUTPUT_PRESETS = {"p_edit": "edit", "dz_before": "dz", "std": "uncertainty", "p_ground": "low_confidence"}


def run_batch(after_files: list, out_dir: str | Path, net, spec: RuntimeSpec, *, before_files: list | None = None,
              gsd: float = 0.5, buffer_m: float = 64.0, workers: int = 2, read_opts: dict | None = None,
              overlays: bool = True, predict_kwargs: dict | None = None, max_block_m: float = 2000.0,
              progress: Callable | None = None, log: Callable = print, cancelled: Callable = lambda: False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    needs_before = bool(spec.prior_channel) or any(c.startswith("sem_") or c == "dtm_before"
                                                    for c in spec.cond_channels)
    if needs_before and not before_files:
        raise ValueError("this model needs the lasground_new-classified tiles (before files)")
    jobs, union = plan(after_files, before_files if needs_before else None, gsd, buffer_m, max_block_m)
    for j in jobs:
        if j.missing_before:
            log(f"[warn] {j.name}: no before file for {j.missing_before}; those areas lack lasground inputs")
    G = Grid.from_bounds(*union, gsd)
    predict_kwargs = dict(predict_kwargs or {})
    base_seed = int(predict_kwargs.pop("seed", 0))
    keys = ["dtm"] + (["p_ground"] if spec.kind == "groundiff" else [])
    if spec.kind == "groundiff" and spec.prior_channel and spec.gate_channel == spec.prior_channel:
        keys.append("p_edit")
    if spec.prior_channel:
        keys.append("dz_before")
    if predict_kwargs.get("n_samples", 1) > 1 or predict_kwargs.get("tta"):
        keys.append("std")
    ov_keys = [k for k in keys if k in OUTPUT_PRESETS and not (k == "p_ground" and "p_edit" in keys)]

    crs = None
    tiles_dir = out / "tiles"
    tiles_dir.mkdir(exist_ok=True)
    placed = {k: [] for k in keys}
    ov_names = [f"{k}{sfx}" for k in ov_keys for sfx in ("_overlay", "_overlay_rgb")] if overlays else []
    placed.update({n: [] for n in ov_names})

    summary = {"tiles": [], "grid": G.to_dict(), "gsd": gsd, "buffer_m": buffer_m}
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        pending = {}
        nxt = 0

        def submit_upto(limit):
            nonlocal nxt
            while nxt < len(jobs) and len(pending) < limit:
                pending[nxt] = ex.submit(prepare_job, jobs[nxt], gsd, read_opts)
                nxt += 1

        submit_upto(max(1, workers) + 1)
        for idx, job in enumerate(jobs):
            if cancelled():
                for f in pending.values():
                    f.cancel()
                break
            arrs, grid, info = pending.pop(idx).result()
            submit_upto(max(1, workers) + 1)
            rec = {"name": job.name, **{k: v for k, v in info.items() if k != "crs_wkt"}}
            if not arrs:
                rec["skipped"] = "no points"
                summary["tiles"].append(rec)
                continue
            crs = crs or info["crs_wkt"]
            t1 = time.time()
            res = predict_scene(arrs, spec, net, seed=base_seed + idx, **predict_kwargs)
            rec["predict_s"] = round(time.time() - t1, 1)
            # crop the job's core out of its buffered grid; place it on the mosaic grid
            c0 = int(round((job.core[0] - grid.xmin) / gsd))
            r0 = int(round((grid.ymax - job.core[3]) / gsd))
            w = int(round((job.core[2] - job.core[0]) / gsd))
            h = int(round((job.core[3] - job.core[1]) / gsd))
            gc = int(round((job.core[0] - G.xmin) / gsd))
            gr = int(round((G.ymax - job.core[3]) / gsd))
            geo = (job.core[0], job.core[3], gsd, info["crs_wkt"])
            for k in keys:
                if k in res:
                    p = tiles_dir / f"{job.name}_{k}.tif"
                    write_geotiff(p, res[k][r0:r0 + h, c0:c0 + w], *geo)
                    placed[k].append((p, gr, gc, h, w))
            if overlays:
                for k in ov_keys:
                    if k in res:
                        rgba = render_rgba(res[k][r0:r0 + h, c0:c0 + w], OUTPUT_PRESETS[k])
                        for sfx, alpha in (("_overlay", True), ("_overlay_rgb", False)):
                            p = tiles_dir / f"{job.name}_{k}{sfx}.tif"
                            write_rgba_geotiff(p, rgba, *geo, alpha=alpha)
                            placed[f"{k}{sfx}"].append((p, gr, gc, h, w))
            summary["tiles"].append(rec)
            log(f"{job.name}: {info['n_points']} pts, read {info.get('read_s')} s, predict {rec['predict_s']} s")
            if progress:
                progress((idx + 1) / len(jobs))

    # mosaics: VRT index over the tile files, then one compressed GeoTIFF each
    outputs = {}
    for name, items in placed.items():
        if not items:
            continue
        is_rgb = name.endswith("_overlay") or name.endswith("_overlay_rgb")
        if is_rgb:
            rgba = name.endswith("_overlay")
            vrt = build_vrt(out / f"{name}.vrt", items, G.width, G.height, G.xmin, G.ymax, gsd, crs,
                            bands=4 if rgba else 3, dtype="Byte", nodata=None if rgba else 0,
                            colorinterp=["Red", "Green", "Blue", "Alpha"] if rgba else ["Red", "Green", "Blue"])
            tif = out / f"{name}.tif"
            vrt_to_geotiff(vrt, tif, rgba=rgba)
            _write_sidecars(tif, G.xmin, G.ymax, gsd, crs)
        else:
            vrt = build_vrt(out / f"{name}.vrt", items, G.width, G.height, G.xmin, G.ymax, gsd, crs)
            tif = out / f"{name}.tif"
            vrt_to_geotiff(vrt, tif)
            preset = OUTPUT_PRESETS.get(name)
            if preset and not PRESETS[preset].get("invert") and not PRESETS[preset].get("abs"):
                tif.with_suffix(".qml").write_text(qml_style(preset))
        outputs[name] = str(tif)
    summary["outputs"] = outputs
    summary["seconds"] = round(time.time() - t_start, 1)
    (out / "batch_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint")
    src.add_argument("--onnx")
    ap.add_argument("--after", nargs="+", required=True, help="EA/LP360 tiles (LAS/LAZ/COPC)")
    ap.add_argument("--before", nargs="*", default=[], help="the same tiles classified by lasground_new")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gsd", type=float, default=0.5)
    ap.add_argument("--buffer", type=float, default=64.0, help="metres of neighbouring points around each tile")
    ap.add_argument("--workers", type=int, default=2, help="tiles read/rasterised in parallel")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--blend", choices=["min", "linear", "mean"], default="min")
    ap.add_argument("--prior", choices=["auto", "global", "channel", "none"], default="auto")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--drop-overlap", action="store_true")
    ap.add_argument("--drop-synthetic", action="store_true")
    ap.add_argument("--no-overlays", action="store_true")
    a = ap.parse_args(argv)
    from .infer import load_net
    net, spec = load_net(a.checkpoint, a.onnx, a.device)
    s = run_batch(a.after, a.out, net, spec, before_files=a.before, gsd=a.gsd, buffer_m=a.buffer,
                  workers=a.workers, overlays=not a.no_overlays,
                  read_opts={"drop_overlap": a.drop_overlap, "drop_synthetic": a.drop_synthetic},
                  predict_kwargs={"blend": a.blend, "prior": a.prior, "n_samples": a.samples, "tta": a.tta,
                                  "batch_size": a.batch_size})
    print(json.dumps({k: v for k, v in s.items() if k != "tiles"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

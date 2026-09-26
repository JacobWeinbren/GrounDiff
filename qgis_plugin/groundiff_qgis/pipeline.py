"""QGIS-independent logic of the plugin (testable without QGIS).

Inputs:
  * point-cloud tiles (any number): the EA/LP360 LAS/LAZ/COPC tiles and, for
    before -> after models, the same tiles re-classified by lasground_new
    (paired by file name). Processed with neighbour buffers into one set of
    mosaic GeoTIFFs in an output folder (core/batch.py);
  * rasters: one GeoTIFF per channel name, all on one grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .core.io_raster import read_geotiff, write_geotiff
from .core.overlay import write_overlays
from .core.runtime import RuntimeSpec, predict_scene

OVERLAY_PRESET = {"p_edit": "edit", "dz_before": "dz", "std": "uncertainty"}


def load_spec(onnx_path: str) -> RuntimeSpec:
    js = Path(onnx_path).with_suffix(".json")
    if not js.exists():
        raise FileNotFoundError(f"{js} not found: export the model with `python -m groundiff.export`")
    return RuntimeSpec.from_json(js)


def rasters_from_files(paths: dict) -> tuple[dict, dict]:
    arrs, info = {}, None
    for name, p in paths.items():
        a, i = read_geotiff(p)
        if info is None:
            info = i
        elif (a.shape != next(iter(arrs.values())).shape or abs(i["gsd"] - info["gsd"]) > 1e-9
              or abs(i["xmin"] - info["xmin"]) > 1e-6 or abs(i["ymax"] - info["ymax"]) > 1e-6):
            raise ValueError(f"raster {name} ({p}) is not on the same grid as the others")
        arrs[name] = a
    return arrs, info


def rasters_from_points(after_laz: str, before_laz: str | None, gsd: float,
                        before_ground_classes=(2,), read_opts: dict | None = None) -> tuple[dict, dict]:
    """One tile, no neighbour buffer (use run_tiles for several tiles)."""
    from .core.data.laz import NOISE_CLASSES, read_points
    from .core.data.rasterise import Grid, build_rasters

    read_opts = read_opts or {}
    pts = read_points(after_laz, drop_classes=NOISE_CLASSES, **read_opts)
    bp = read_points(before_laz, drop_classes=NOISE_CLASSES, **read_opts) if before_laz else None
    grid = Grid.from_bounds(pts.x.min(), pts.y.min(), pts.x.max(), pts.y.max(), gsd)
    arrs = build_rasters(grid, pts, bp, before_ground_classes=before_ground_classes, with_target=False)
    return ({k: v.astype(np.float64) for k, v in arrs.items()},
            {"xmin": grid.xmin, "ymax": grid.ymax, "gsd": grid.gsd, "crs_wkt": pts.crs_wkt})


def run(onnx_path: str, arrs: dict, info: dict, outputs: dict, providers: list | None = None,
        stride: int | None = None, blend: str = "min", prior: str = "auto", n_samples: int = 1,
        tta: bool = False, batch_size: int = 8, seed: int = 0, progress=None, crs_wkt: str | None = None,
        overlays: bool = True) -> dict:
    """outputs: {"dtm", "p_ground", "p_edit", "std", "dz_before"} -> path; empty or missing are skipped."""
    from .core.backends import OnnxNet

    spec = load_spec(onnx_path)
    missing = [c for c in spec.needed_channels if c not in arrs]
    if missing:
        raise ValueError(f"the model needs these inputs, which were not provided: {missing}")
    net = OnnxNet(onnx_path, providers)
    res = predict_scene(arrs, spec, net, stride=stride, blend=blend, prior=prior, n_samples=n_samples,
                        tta=tta, batch_size=batch_size, seed=seed, progress=progress)
    written = {}
    crs = crs_wkt or info.get("crs_wkt")
    for key, path in outputs.items():
        if path and key in res:
            write_geotiff(path, res[key], info["xmin"], info["ymax"], info["gsd"], crs)
            written[key] = path
            preset = OVERLAY_PRESET.get(key) or ("low_confidence" if key == "p_ground" and "p_edit" not in res else None)
            if overlays and preset:
                stem = Path(path).with_suffix("")
                written[f"{key}_overlays"] = [str(p) for p in
                                              write_overlays(res[key], stem, preset, info["xmin"], info["ymax"],
                                                             info["gsd"], crs)]
    written["providers"] = net.providers
    return written


def inspect_report(path: str, compare_to: str | None = None) -> str:
    from .core.data.lasinspect import compare, inspect
    lines = []
    reps = [inspect(path)] + ([inspect(compare_to)] if compare_to else [])
    for r in reps:
        lines.append(f"== {r['file']}")
        lines += [f"  {k}: {v}" for k, v in r.items() if k not in ("file", "warnings")]
        lines += [f"  WARNING: {w}" for w in r["warnings"]]
    if compare_to:
        lines.append("== differences (first | second)")
        lines += [f"  {k}: {v}" for k, v in compare(reps[0], reps[1]).items()]
    return "\n".join(lines)


def run_tiles(onnx_path: str, after_files: list, out_dir: str, before_files: list | None = None,
              providers: list | None = None, gsd: float = 0.5, buffer_m: float = 64.0, workers: int = 2,
              read_opts: dict | None = None, overlays: bool = True, predict_kwargs: dict | None = None,
              progress=None, log=print, cancelled=lambda: False) -> dict:
    from .core.backends import OnnxNet
    from .core.batch import run_batch

    exts = (".las", ".laz")
    after_files = [f for f in after_files if str(f).lower().endswith(exts)]
    before_files = [f for f in (before_files or []) if str(f).lower().endswith(exts)]
    if not after_files:
        raise ValueError("no .las/.laz files selected")
    spec = load_spec(onnx_path)
    net = OnnxNet(onnx_path, providers)
    log(f"ONNX Runtime providers: {net.providers}")
    return run_batch(after_files, out_dir, net, spec, before_files=before_files, gsd=gsd, buffer_m=buffer_m,
                     workers=workers, read_opts=read_opts, overlays=overlays, predict_kwargs=predict_kwargs,
                     progress=progress, log=log, cancelled=cancelled)

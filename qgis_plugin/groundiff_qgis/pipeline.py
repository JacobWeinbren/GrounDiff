"""QGIS-independent logic of the plugin (testable without QGIS).

Inputs:
  * point-cloud tiles (any number) as lasground_new wrote them (default
    settings: classes 1/2), processed with neighbour buffers into one set of
    mosaic GeoTIFFs plus a priority list in an output folder (core/batch.py);
  * rasters: one GeoTIFF per channel name, all on one grid (e.g. written by
    `python -m groundiff.data.preprocess --geotiff`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .core.io_raster import read_geotiff, write_geotiff
from .core.overlay import write_overlays
from .core.runtime import RuntimeSpec, lattice_anchor, predict_scene

OVERLAY_PRESET = {"p_edit": "edit", "dz_before": "dz", "std": "uncertainty"}


def load_spec(onnx_path: str) -> RuntimeSpec:
    js = Path(onnx_path).with_suffix(".json")
    if not js.exists():
        raise FileNotFoundError(f"{js} not found: export the model with `python -m groundiff.export`")
    return RuntimeSpec.from_json(js)


def producible(spec: RuntimeSpec, n_samples: int = 1, tta: bool = False, has_before: bool = True) -> list[str]:
    """Outputs the model can give. For DSM -> DTM models dz_before and p_edit
    need the lasground_new DTM (a dtm_before raster, or lasground_new tiles)."""
    from .core.batch import output_keys
    keys = output_keys(spec, {"n_samples": n_samples, "tta": tta})
    if not has_before and not spec.needs_before:
        keys = [k for k in keys if k not in ("dz_before", "p_edit")]
    return keys


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


def rasters_from_points(tile: str, gsd: float, lasground: bool = True, before_ground_classes=(2,),
                        read_opts: dict | None = None) -> tuple[dict, dict]:
    """One tile, no neighbour buffer (use run_tiles for several tiles)."""
    from .core.data.laz import read_points
    from .core.data.rasterise import Grid, input_rasters
    from .core.io_raster import crs_wkt_from_epsg

    pts = read_points(tile, **(read_opts or {}))
    grid = Grid.from_bounds(pts.x.min(), pts.y.min(), pts.x.max(), pts.y.max(), gsd)
    arrs = input_rasters(grid, pts, lasground, before_ground_classes)
    return ({k: v.astype(np.float64) for k, v in arrs.items()},
            {"xmin": grid.xmin, "ymax": grid.ymax, "gsd": grid.gsd,
             "crs_wkt": pts.crs_wkt or crs_wkt_from_epsg(27700)})


def run(onnx_path: str, arrs: dict, info: dict, outputs: dict, providers: list | None = None,
        stride: int | None = None, blend: str = "linear", prior: str = "auto", n_samples: int = 1,
        tta: bool = False, batch_size: int = 8, seed: int = 0, progress=None, crs_wkt: str | None = None,
        overlays: bool = True) -> dict:
    """outputs: {"dtm", "p_ground", "p_edit", "std", "dz_before"} -> path; empty,
    missing or not producible by this model are skipped."""
    from .core.backends import OnnxNet

    spec = load_spec(onnx_path)
    missing = [c for c in spec.needed_channels if c not in arrs]
    if missing:
        raise ValueError(f"the model needs these inputs, which were not provided: {missing}")
    net = OnnxNet(onnx_path, providers)
    # same tile lattice and per-tile noise as the tile mode (batch), so both give the same values
    res = predict_scene(arrs, spec, net, stride=stride, blend=blend, prior=prior, n_samples=n_samples,
                        tta=tta, batch_size=batch_size, seed=seed, progress=progress, gsd=info.get("gsd"),
                        anchor=lattice_anchor(info["xmin"], info["ymax"], info["gsd"]))
    written = {}
    crs = crs_wkt or info.get("crs_wkt")
    for key, path in outputs.items():
        if path and key in res:
            write_geotiff(path, res[key], info["xmin"], info["ymax"], info["gsd"], crs)
            written[key] = path
            preset = OVERLAY_PRESET.get(key)
            if overlays and preset:
                stem = Path(path).with_suffix("")
                written[f"{key}_overlays"] = [str(p) for p in
                                              write_overlays(res[key], stem, preset, info["xmin"], info["ymax"],
                                                             info["gsd"], crs)]
    written["providers"] = net.providers
    written["warning"] = net.warning
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


def run_tiles(onnx_path: str, tiles: list, out_dir: str, providers: list | None = None, gsd: float | None = None,
              buffer_m: float | None = None, workers: int = 2, read_opts: dict | None = None, overlays: bool = True,
              predict_kwargs: dict | None = None, block_m: float = 100.0, progress=None, log=print,
              cancelled=lambda: False) -> dict:
    """tiles: LAS/LAZ/COPC files as lasground_new wrote them. gsd / buffer
    None: as trained / one network tile + 32 m."""
    from .core.backends import OnnxNet
    from .core.batch import run_batch

    tiles = [f for f in tiles if str(f).lower().endswith((".las", ".laz"))]
    if not tiles:
        raise ValueError("no .las/.laz files selected")
    spec = load_spec(onnx_path)
    net = OnnxNet(onnx_path, providers)
    log(f"ONNX Runtime providers: {net.providers}")
    if net.warning:
        log(f"[warn] {net.warning}")
    s = run_batch(tiles, out_dir, net, spec, gsd=gsd, buffer_m=buffer_m, workers=workers, read_opts=read_opts,
                  overlays=overlays, predict_kwargs=predict_kwargs, block_m=block_m, progress=progress, log=log,
                  cancelled=cancelled)
    s["providers"] = net.providers
    return s

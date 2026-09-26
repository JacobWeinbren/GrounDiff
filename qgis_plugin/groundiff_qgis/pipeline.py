"""QGIS-independent logic of the plugin (testable without QGIS).

Two ways to supply inputs:
  * rasters: one GeoTIFF per channel name (as written by `groundiff.infer`
    or produced in QGIS), all on the same grid;
  * point clouds: the EA LAZ/COPC tile and, for before -> after models, the
    same tile re-classified by lasground_new; rasters are built exactly as
    in training (core/data/rasterise.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .core.io_raster import read_geotiff, write_geotiff
from .core.runtime import RuntimeSpec, predict_scene


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
                        before_ground_classes=(2,)) -> tuple[dict, dict]:
    from .core.data.laz import NOISE_CLASSES, read_points
    from .core.data.rasterise import Grid, class_mode_onehot, rasterise_points, tin_dtm

    pts = read_points(after_laz, drop_classes=NOISE_CLASSES)
    grid = Grid.from_bounds(pts.x.min(), pts.y.min(), pts.x.max(), pts.y.max(), gsd)
    arrs = {k: v.astype(np.float64) for k, v in
            rasterise_points(grid, pts.x, pts.y, pts.z, pts.return_number, pts.number_of_returns).items()}
    if before_laz:
        bp = read_points(before_laz, drop_classes=NOISE_CLASSES)
        bg = np.isin(bp.cls, np.asarray(before_ground_classes, np.uint8))
        dtm_b, _ = tin_dtm(grid, bp.x[bg], bp.y[bg], bp.z[bg])
        arrs["dtm_before"] = dtm_b.astype(np.float64)
        sem = class_mode_onehot(grid, bp.x, bp.y, bg)
        arrs["sem_ground"], arrs["sem_nonground"] = sem[0], sem[1]
    return arrs, {"xmin": grid.xmin, "ymax": grid.ymax, "gsd": grid.gsd, "crs_wkt": pts.crs_wkt}


def run(onnx_path: str, arrs: dict, info: dict, outputs: dict, providers: list | None = None,
        stride: int | None = None, blend: str = "min", prior: str = "auto", n_samples: int = 1,
        tta: bool = False, batch_size: int = 8, seed: int = 0, progress=None, crs_wkt: str | None = None) -> dict:
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
    written["providers"] = net.providers
    return written

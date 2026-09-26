"""The QGIS plugin's logic, run without QGIS (the thin QGIS wrapper is only
compiled here)."""
import importlib
import json
import py_compile
import sys

import numpy as np
import pytest

from groundiff.export import export
from groundiff.io_raster import read_geotiff, write_geotiff
from tests.test_runtime import trained  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def plugin(tmp_path_factory):
    sys.path.insert(0, "tools")
    from build_qgis_plugin import build
    dest = tmp_path_factory.mktemp("plugin")
    folder = build(dest)
    sys.path.insert(0, str(dest))
    mod = importlib.import_module("groundiff_qgis.pipeline")
    yield folder, mod
    sys.path.remove(str(dest))


def test_wrappers_compile(plugin):
    folder, _ = plugin
    for f in ("__init__.py", "plugin.py", "provider.py", "algorithm.py"):
        py_compile.compile(str(folder / f), doraise=True)
    assert "hasProcessingProvider=yes" in (folder / "metadata.txt").read_text()


def test_plugin_runs_from_rasters_and_points(trained, plugin, tmp_path):  # noqa: F811
    root, model, cfg = trained
    _, pipe = plugin
    onnx_path, _ = export(root / "run" / "last.pt", tmp_path / "model")
    spec = pipe.load_spec(str(onnx_path))
    # raster input: write the scene's channels as GeoTIFFs, as a QGIS user would have them
    sd = root / "scenes" / "S2"
    meta = json.loads((sd / "meta.json").read_text())
    g = meta["grid"]
    paths = {}
    for ch in spec.needed_channels:
        p = tmp_path / f"{ch}.tif"
        write_geotiff(p, np.load(sd / f"{ch}.npy"), g["xmin"], g["ymax"], g["gsd"], meta["crs_wkt"])
        paths[ch] = str(p)
    arrs, info = pipe.rasters_from_files(paths)
    outs = {"dtm": str(tmp_path / "dtm.tif"), "p_ground": str(tmp_path / "pg.tif"), "std": "",
            "dz_before": str(tmp_path / "dz.tif")}
    written = pipe.run(str(onnx_path), arrs, info, outs, providers=["CPUExecutionProvider"], batch_size=4)
    dtm, dinfo = read_geotiff(written["dtm"])
    assert dtm.shape == (g["height"], g["width"]) and dinfo["xmin"] == g["xmin"]
    assert "std" not in written
    # point-cloud input gives the same rasters as preprocessing, so the same DTM
    arrs2, info2 = pipe.rasters_from_points(meta["after_file"], meta["before_file"], g["gsd"])
    out2 = {"dtm": str(tmp_path / "dtm2.tif")}
    pipe.run(str(onnx_path), arrs2, info2, out2, providers=["CPUExecutionProvider"], batch_size=4)
    dtm2, _ = read_geotiff(out2["dtm"])
    ok = np.isfinite(dtm) & np.isfinite(dtm2)
    assert ok.mean() > 0.9 and np.abs(dtm[ok] - dtm2[ok]).max() < 1e-3
    with pytest.raises(ValueError):
        pipe.run(str(onnx_path), {"dsm_max": arrs["dsm_max"]}, info, outs)

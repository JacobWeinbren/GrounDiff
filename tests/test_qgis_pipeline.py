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
    can = pipe.producible(spec)
    edit = spec.gate_channel == spec.prior_channel
    assert set(can) == ({"dtm", "p_edit", "dz_before"} if edit else {"dtm", "p_ground", "p_edit", "dz_before"})
    outs = {"dtm": str(tmp_path / "dtm.tif"), "p_edit": str(tmp_path / "pe.tif"), "std": "",
            "dz_before": str(tmp_path / "dz.tif")}
    written = pipe.run(str(onnx_path), arrs, info, outs, providers=["CPUExecutionProvider"], batch_size=4,
                       blend="linear")
    dtm, dinfo = read_geotiff(written["dtm"])
    assert dtm.shape == (g["height"], g["width"]) and dinfo["xmin"] == g["xmin"]
    assert "std" not in written and (tmp_path / "dz.qml").exists()
    assert set(k for k in written if k in ("dtm", "p_edit", "dz_before", "std", "p_ground")) <= set(can)
    # tile mode on the same point cloud (single tile, no neighbours) gives the same DTM
    s = pipe.run_tiles(str(onnx_path), [meta["before_file"]], str(tmp_path / "tiles_out"),
                       providers=["CPUExecutionProvider"], buffer_m=0.0, workers=1,
                       predict_kwargs={"batch_size": 4, "blend": "linear"})
    dtm2, info2 = read_geotiff(s["outputs"]["dtm"])
    # same grid origin => compare the overlapping window
    dc = int(round((info2["xmin"] - dinfo["xmin"]) / g["gsd"]))
    dr = int(round((dinfo["ymax"] - info2["ymax"]) / g["gsd"]))
    sub = dtm[dr:dr + dtm2.shape[0], dc:dc + dtm2.shape[1]]
    ok = np.isfinite(sub) & np.isfinite(dtm2[:sub.shape[0], :sub.shape[1]])
    assert ok.mean() > 0.9
    assert np.abs(sub[ok] - dtm2[:sub.shape[0], :sub.shape[1]][ok]).max() < 1e-3
    assert "p_edit_overlay" not in s["outputs"] or s["outputs"]["p_edit_overlay"].endswith(".tif")
    with pytest.raises(ValueError):
        pipe.run(str(onnx_path), {"dsm_max": arrs["dsm_max"]}, info, outs)


def test_plugin_inspect(trained, plugin):  # noqa: F811
    root, _, _ = trained
    _, pipe = plugin
    import json as _json
    meta = _json.loads((root / "scenes" / "S2" / "meta.json").read_text())
    rep = pipe.inspect_report(meta["after_file"], meta["before_file"])
    assert "point_format" in rep and "differences" in rep


@pytest.mark.parametrize("new_enums", [True, False])
def test_algorithms_run_under_stub_qgis(trained, plugin, tmp_path, new_enums):  # noqa: F811
    """Runs the real Processing algorithm classes against a stand-in qgis.core
    (QGIS 3.22-3.34 and 3.36+/4.x enum styles)."""
    from tests import qgis_stub
    qgis_stub.install(new_enums)
    folder, _ = plugin
    for m in [m for m in list(sys.modules) if m.startswith("groundiff_qgis.algorithm")]:
        del sys.modules[m]
    alg_mod = importlib.import_module("groundiff_qgis.algorithm")
    root, _, _ = trained
    onnx_path, _ = export(root / "run" / "last.pt", tmp_path / "model")
    meta = json.loads((root / "scenes" / "S2" / "meta.json").read_text())

    # tiles
    alg = alg_mod.PredictTilesAlgorithm().createInstance()
    alg.initAlgorithm()
    ctx, fb = qgis_stub.install(new_enums).QgsProcessingContext(), qgis_stub.Feedback()
    params = {"MODEL": str(onnx_path), "TILES": [meta["before_file"], str(tmp_path / "notes.txt")],
              "OUTPUT_FOLDER": str(tmp_path / "out"), "BACKEND": 4, "BATCH": 4, "WORKERS": 1}
    res = alg.processAlgorithm(params, ctx, fb)
    assert res["OUTPUT_FOLDER"] == str(tmp_path / "out")
    assert (tmp_path / "out" / "dtm.tif").exists() and (tmp_path / "out" / "priority.geojson").exists()
    loaded = {d.name: d for d in ctx.to_load.values()}
    assert "GrounDiff DTM" in loaded and "Edit priority blocks" in loaded
    styled = [d for d in loaded.values() if d.post is not None]
    assert styled and all(d.post.qml.endswith(".qml") for d in styled)

    # rasters: std needs several samples; with SAMPLES=1 it is not producible -> warning, no output
    alg = alg_mod.PredictRastersAlgorithm().createInstance()
    alg.initAlgorithm()
    ctx, fb = qgis_stub.install(new_enums).QgsProcessingContext(), qgis_stub.Feedback()
    spec = importlib.import_module("groundiff_qgis.pipeline").load_spec(str(onnx_path))
    sd, g = root / "scenes" / "S2", meta["grid"]
    params = {"MODEL": str(onnx_path), "BACKEND": 4, "BATCH": 4, "SAMPLES": 1,
              "DTM": str(tmp_path / "r_dtm.tif"), "P_EDIT": str(tmp_path / "r_pe.tif"),
              "DZ_BEFORE": str(tmp_path / "r_dz.tif"), "STD": str(tmp_path / "r_std.tif")}
    for ch in spec.needed_channels:
        p = tmp_path / f"in_{ch}.tif"
        write_geotiff(p, np.load(sd / f"{ch}.npy"), g["xmin"], g["ymax"], g["gsd"], meta["crs_wkt"])
        params[ch.upper()] = str(p)
    res = alg.processAlgorithm(params, ctx, fb)
    assert "DTM" in res and "P_EDIT" in res and "STD" not in res
    assert any("std" in w for w in fb.warnings) and str(tmp_path / "r_std.tif") not in ctx.to_load
    assert ctx.to_load[str(tmp_path / "r_dz.tif")].post is not None          # dz styled with its .qml

    # inspect
    alg = alg_mod.InspectLasAlgorithm().createInstance()
    alg.initAlgorithm()
    fb = qgis_stub.Feedback()
    alg.processAlgorithm({"FILE": meta["before_file"]}, ctx, fb)
    assert any("point_format" in m for m in fb.info)
    for m in ("qgis", "qgis.core"):
        sys.modules.pop(m, None)

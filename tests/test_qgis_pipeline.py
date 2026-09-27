"""The QGIS plugin's logic, run without QGIS (the thin QGIS wrapper is only
compiled here)."""
import importlib
import os
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
    params = {"MODEL": str(onnx_path), "LAYERS": [meta["before_file"]], "TILES": [str(tmp_path / "notes.txt")],
              "OUTPUT_FOLDER": str(tmp_path / "out"), "BACKEND": 4, "BATCH": 4, "WORKERS": 1}
    res = alg.processAlgorithm(params, ctx, fb)
    assert res["OUTPUT_FOLDER"] == str(tmp_path / "out")
    assert (tmp_path / "out" / "dtm.tif").exists() and (tmp_path / "out" / "priority.geojson").exists()
    loaded = {d.name: d for d in ctx.to_load.values()}
    assert "GrounDiff DTM" in loaded and "Edit priority blocks" in loaded
    styled = [d for d in loaded.values() if d.post is not None]
    assert styled and all(d.post.qml.endswith(".qml") for d in styled)

    assert alg.params["BACKEND"].flags() and not alg.params["TILES"].flags()     # technical ones are 'Advanced'
    # second run: no model given -> the remembered one; no folder -> a temporary one
    ctx, fb = qgis_stub.install(new_enums).QgsProcessingContext(), qgis_stub.Feedback()
    # as QGIS 3.44 on macOS passes them: a 'pdal://' layer source and [''] for the empty file box
    res2 = alg.processAlgorithm({"LAYERS": ["pdal://" + meta["before_file"]], "TILES": [""], "BACKEND": 4,
                                 "BATCH": 4, "SAMPLES": 1, "OUTPUT_FOLDER": "TEMPORARY_OUTPUT"}, ctx, fb)
    assert os.path.exists(os.path.join(res2["OUTPUT_FOLDER"], "dtm.tif"))
    assert fb.progress[-1] == 100 and max(fb.progress) <= 100
    assert any("reading points" in t for t in fb.texts) and any("model" in t for t in fb.texts)
    assert any("Joining tiles" in t for t in fb.texts) and fb.texts[-1].startswith("Done")
    assert alg_mod.layer_file("file:///C:/data/a%20b.laz") == "C:/data/a b.laz"

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

    # device test: runs the plugin's own copy in a child process and remembers the choice
    os.environ["GROUNDIFF_CACHE"] = str(tmp_path / "cache")
    alg = alg_mod.TestDevicesAlgorithm().createInstance()
    alg.initAlgorithm()
    fb = qgis_stub.Feedback()
    alg.processAlgorithm({"MODEL": str(onnx_path), "BATCH": "2 4"}, ctx, fb)
    assert any(m.startswith("cpu, batch 2: ") and "per tile-step" in m for m in fb.info), fb.info
    assert (tmp_path / "cache" / "devices.json").exists()
    del os.environ["GROUNDIFF_CACHE"]
    for m in ("qgis", "qgis.core"):
        sys.modules.pop(m, None)


def test_spec_embedded_in_onnx_is_enough(trained, plugin, tmp_path):  # noqa: F811
    root, _, _ = trained
    _, pipe = plugin
    onnx_path, json_path = export(root / "run" / "last.pt", tmp_path / "m")
    json_path.unlink()                                    # only the .onnx is copied to the other machine
    spec = pipe.load_spec(str(onnx_path))
    assert spec.cond_channels and spec.tile > 0


def test_dependency_installer_commands(plugin, tmp_path, monkeypatch):
    folder, _ = plugin
    deps = importlib.import_module("groundiff_qgis.deps")
    cmds = deps.pip_commands(["onnxruntime", "laspy", "lazrs"], tmp_path)
    flat = [" ".join(c) for c in cmds]
    assert all("--target" in c and str(tmp_path) in c for c in flat)
    assert any("--no-deps" in c and "onnxruntime" in c and "laspy" in c for c in flat)
    assert not any("numpy" in c.split() for c in flat)                     # QGIS's numpy is never replaced
    assert "lazrs" in flat[-1] and "--no-deps" not in flat[-1]
    assert deps.missing() == [] or all(isinstance(m, str) for m in deps.missing())
    assert os.path.exists(deps.python_exe())

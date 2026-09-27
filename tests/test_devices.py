import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")


@pytest.fixture
def tiny_onnx(tmp_path):
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Conv2d(3, 2, 3, padding=1)

        def forward(self, x, gamma):
            return self.c(x) * gamma[:, None, None, None]

    p = tmp_path / "tiny.onnx"
    torch.onnx.export(Net().eval(), (torch.randn(2, 3, 16, 16), torch.rand(2)), str(p), input_names=["x", "gamma"],
                      output_names=["out"], dynamic_axes={"x": {0: "batch"}, "gamma": {0: "batch"},
                                                          "out": {0: "batch"}}, opset_version=17, dynamo=False)
    return p


def test_fixed_batch_pads_and_splits(tiny_onnx):
    from groundiff.backends import OnnxNet
    a = OnnxNet(tiny_onnx, "cpu")
    b = OnnxNet(tiny_onnx, "cpu")
    assert a.fixed_batch is None                       # only accelerators get a fixed batch
    b.fixed_batch = 4
    x = np.random.default_rng(0).standard_normal((7, 3, 16, 16)).astype(np.float32)
    g = np.linspace(0.1, 0.9, 7).astype(np.float32)
    assert np.array_equal(a(x, g), b(x, g))
    assert np.array_equal(a(x[:1], g[:1]), b(x[:1], g[:1]))


def test_speedtest_remembers_best_device(tiny_onnx, tmp_path, monkeypatch):
    from groundiff import speedtest
    from groundiff.backends import auto_providers
    monkeypatch.setenv("GROUNDIFF_CACHE", str(tmp_path / "cache"))
    logs = []
    res = speedtest.run(str(tiny_onnx), devices=["cpu"], batch=2, reps=1, log=logs.append)
    assert res[0]["ok"] and res[0]["max_diff"] == 0.0 and res[0]["s_per_tile_step"] > 0
    assert speedtest.best_device(tiny_onnx) == "cpu"
    state = json.loads((tmp_path / "cache" / "devices.json").read_text())
    assert list(state.values())[0]["best"] == "cpu"
    assert auto_providers(["CPUExecutionProvider"], tiny_onnx) == "cpu"            # the tested device
    assert any("fastest with results matching the CPU" in m for m in logs)


def test_installer_picks_cuda_build_with_nvidia(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path
    f = Path(__file__).resolve().parents[1] / "qgis_plugin" / "groundiff_qgis" / "deps.py"
    spec = importlib.util.spec_from_file_location("_gd_deps_test", f)    # not the package: other tests
    deps = importlib.util.module_from_spec(spec)                         # import the built plugin
    spec.loader.exec_module(deps)
    monkeypatch.setattr(deps, "has_nvidia", lambda: True)
    assert deps.ort_package() == "onnxruntime-gpu[cuda,cudnn]"
    monkeypatch.setattr(deps, "python_exe", lambda: "python")
    cmds = deps.pip_commands(["onnxruntime-gpu[cuda,cudnn]", "laspy"], tmp_path)
    gpu = [c for c in cmds if "onnxruntime-gpu[cuda,cudnn]" in c]
    assert gpu and "--no-deps" not in gpu[0]          # the NVIDIA libraries come as its dependencies
    assert any("--no-deps" in c and "laspy" in c for c in cmds)
    monkeypatch.setattr(deps, "has_nvidia", lambda: False)
    assert deps.ort_package() in ("onnxruntime", "onnxruntime-directml")


def test_speedtest_batches_and_best_setting(tiny_onnx, tmp_path, monkeypatch):
    from groundiff import speedtest
    monkeypatch.setenv("GROUNDIFF_CACHE", str(tmp_path / "cache"))
    res = speedtest.run(str(tiny_onnx), devices=["cpu"], batch=[2, 4], reps=1, log=lambda m: None)
    assert [r["batch"] for r in res] == [2]            # the CPU reference runs once
    assert speedtest.best_setting(tiny_onnx) == ("cpu", 2)


def test_fp16_export_close_to_fp32(tmp_path):
    """The mixed-precision export runs and stays close to float32 (the speed test applies the limit)."""
    import onnxruntime as ort
    from groundiff.config import load_config
    from groundiff.export import _Denoiser, _HalfDenoiser
    from groundiff.models.build import build_model
    cfg = load_config("configs/no_lastools.json")
    cfg.model.inner_channel = 32
    cfg.model.channel_mults = [1, 2]
    torch.manual_seed(0)
    m = build_model(cfg).eval()
    for p in m.denoiser.parameters():               # zero-initialised output head: give it weights
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.02)
    x = torch.clamp(torch.randn(2, 7, 64, 64), -1, 1)
    g = torch.tensor([0.2, 0.8])
    with torch.no_grad():
        ref = _Denoiser(m.denoiser).eval()(x, g).numpy()
        f = tmp_path / "h.onnx"
        torch.onnx.export(_HalfDenoiser(m.denoiser).eval(), (x, g), str(f), input_names=["x", "gamma"],
                          output_names=["out"], opset_version=18, dynamo=False)
    out = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy(),
                                                                                       "gamma": g.numpy()})[0]
    assert out.dtype == np.float32 and out.shape == ref.shape
    assert 0 < np.abs(out - ref).max() < 1e-2


def test_speedtest_reference_is_the_main_model(tiny_onnx, tmp_path, monkeypatch):
    """--also files are compared with the main model on the CPU, not with their own CPU run."""
    import onnx
    from groundiff import speedtest
    monkeypatch.setenv("GROUNDIFF_CACHE", str(tmp_path / "cache"))
    m = onnx.load(str(tiny_onnx))
    for t in m.graph.initializer:                    # a copy with different weights
        if t.name.endswith("weight"):
            arr = onnx.numpy_helper.to_array(t) * 1.5
            t.CopyFrom(onnx.numpy_helper.from_array(arr.astype(np.float32), t.name))
    other = tmp_path / "other.onnx"
    onnx.save(m, str(other))
    res = speedtest.run(str(tiny_onnx), devices=["cpu"], batch=2, reps=1, log=lambda s: None, also=[str(other)])
    o = [r for r in res if r["model"] == str(other.resolve())][0]
    assert o["max_diff"] > speedtest.TOLERANCE and not o["ok"]

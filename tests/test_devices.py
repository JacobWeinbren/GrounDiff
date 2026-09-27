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
    assert auto_providers(["CPUExecutionProvider"], tiny_onnx) == ["CPUExecutionProvider"]
    assert any("Fastest" in m for m in logs)


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

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from groundiff.backends import OnnxNet, TorchNet
from groundiff.data.preprocess import process_scene
from groundiff.export import export
from groundiff.infer import capture_at, run_scene
from groundiff.models.build import load_model
from groundiff.runtime import RuntimeSpec, predict_scene, sample
from tests.synthetic import write_scene
from tests.test_train import base_cfg

from groundiff.train import train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("rt")
    for i in range(3):
        after, before = write_scene(root / "laz", name=f"S{i}", seed=10 + i, size=80.0)
        process_scene(after, root / "scenes", before=before, gsd=1.0)
    (root / "split.json").write_text(json.dumps({"train": ["S0", "S1"], "val": ["S2"], "test": ["S2"]}))
    out = root / "run"
    train(base_cfg(root, out, optim={"total_steps": 3, "warmup_steps": 1, "lr": 1e-3}))
    model, cfg, _ = load_model(out / "last.pt")
    # make the network non-trivial so equivalence tests are meaningful
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    return root, model, cfg


def test_numpy_sampler_matches_torch(trained):
    _, model, cfg = trained
    spec = RuntimeSpec.from_config(cfg)
    net = TorchNet(model, "cpu")
    torch.manual_seed(0)
    cond = torch.randn(2, len(spec.cond_channels), 32, 32)
    prior = torch.randn(2, 1, 32, 32)
    for init, t0 in (("dsm_noise", None), ("prior", None), ("prior", 4), ("dsm", None)):
        a, la = model.sample(cond, init=init, prior=prior, t_start=t0, add_noise=False)
        b, lb = sample(net, cond.numpy(), spec, init=init, prior=prior.numpy(), t_start=t0, add_noise=False)
        assert np.allclose(a.numpy(), b, atol=2e-4), init
        assert np.allclose(la.numpy(), lb, atol=2e-3)


def _scene_arrays(root, name, spec):
    sd = root / "scenes" / name
    names = set(spec.needed_channels) | {"gt_dtm", "gt_valid"}
    return {n: np.load(sd / f"{n}.npy") for n in names}


@pytest.mark.parametrize("blend", ["min", "linear", "mean"])
@pytest.mark.parametrize("prior", ["channel", "global", "none"])
def test_predict_scene_modes(trained, blend, prior):
    root, model, cfg = trained
    spec = RuntimeSpec.from_config(cfg)
    arrs = _scene_arrays(root, "S2", spec)
    res = predict_scene(arrs, spec, TorchNet(model, "cpu"), blend=blend, prior=prior, batch_size=4)
    H, W = arrs["dsm_max"].shape
    assert res["dtm"].shape == (H, W)
    data = np.isfinite(arrs["dsm_max"])
    assert np.isfinite(res["dtm"][data]).all()
    assert (res["coverage"] >= 1).all()
    assert np.nanmin(res["p_ground"]) >= 0 and np.nanmax(res["p_ground"]) <= 1
    assert "dz_before" in res


def test_predict_scene_uncertainty(trained):
    root, model, cfg = trained
    spec = RuntimeSpec.from_config(cfg)
    arrs = _scene_arrays(root, "S2", spec)
    res = predict_scene(arrs, spec, TorchNet(model, "cpu"), n_samples=2, tta=True, batch_size=4)
    assert "std" in res and np.nanmax(res["std"]) > 0


def test_onnx_export_and_runtime(trained, tmp_path):
    root, model, cfg = trained
    onnx_path, json_path = export(root / "run" / "last.pt", tmp_path / "m")
    spec = RuntimeSpec.from_json(json_path)
    arrs = _scene_arrays(root, "S2", spec)
    a = predict_scene(arrs, spec, TorchNet(load_model(root / "run" / "last.pt")[0], "cpu"), seed=3, batch_size=4)
    b = predict_scene(arrs, spec, OnnxNet(onnx_path, ["CPUExecutionProvider"]), seed=3, batch_size=4)
    ok = np.isfinite(a["dtm"])
    assert np.abs(a["dtm"][ok] - b["dtm"][ok]).max() < 1e-2


def test_run_scene_outputs(trained, tmp_path):
    root, model, cfg = trained
    spec = RuntimeSpec.from_config(cfg)
    args = SimpleNamespace(stride=None, blend="min", prior="auto", init=None, t_start=None, samples=1,
                           tta=False, batch_size=4, seed=0, block_m=16.0)
    summ = run_scene(root / "scenes" / "S2", TorchNet(model, "cpu"), spec, tmp_path / "S2", args)
    for f in ("dtm.tif", "p_ground.tif", "dz_before.tif", "error.tif", "priority.csv", "metrics.json"):
        assert (tmp_path / "S2" / f).exists(), f
    assert "rmse" in summ["model"] and "rmse" in summ["lasground_new"]
    assert 0.0 <= summ["priority"]["capture_top20pct"] <= 1.0
    from groundiff.io_raster import read_geotiff
    dtm, info = read_geotiff(tmp_path / "S2" / "dtm.tif")
    meta = json.loads((root / "scenes" / "S2" / "meta.json").read_text())
    assert info["xmin"] == meta["grid"]["xmin"] and info["gsd"] == meta["grid"]["gsd"]


def test_capture_at():
    rows = [{"true_edit_volume": v} for v in (10, 5, 0, 0, 0, 0, 0, 0, 0, 0)]
    c = capture_at(rows, fractions=(0.1, 0.2))
    assert c["capture_top10pct"] == pytest.approx(10 / 15) and c["capture_top20pct"] == 1.0

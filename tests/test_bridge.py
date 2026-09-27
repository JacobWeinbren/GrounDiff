"""Residual Diffusion Bridge Model (process="rdbm") and the tile_range input channel."""
import numpy as np
import pytest
import torch

from groundiff.backends import OnnxNet
from groundiff.export import export
from groundiff.models.build import load_model
from groundiff.runtime import RuntimeSpec, predict_scene, sample
from groundiff.schedule import bridge_schedule, bridge_step, bridge_times
from groundiff.train import train
from tests.test_train import base_cfg, scenes  # noqa: F401  (fixture)


def test_bridge_schedule_matches_rdbm_reference():
    """The same numbers as MiliLab/RDBM's rdbm.py (torch, float32), recomputed independently."""
    T, lamb = 100, 1e-4
    th, sg = bridge_schedule(T, lamb)
    import math
    ab = lambda s: math.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2          # noqa: E731
    thetas = torch.tensor([min(1 - ab((i + 1) / T) / ab(i / T), 0.999) for i in range(T)], dtype=torch.float32)
    c0t = thetas.cumsum(0)
    ctT = c0t[-1] - c0t
    ref_th = torch.sinh(ctT) / torch.sinh(c0t[-1])
    ref_sg = torch.sqrt(2 * lamb * torch.sinh(c0t) * torch.sinh(ctT) / torch.sinh(c0t[-1]))
    assert np.allclose(th, ref_th.numpy(), atol=1e-6) and np.allclose(sg, ref_sg.numpy(), atol=1e-6)
    assert th[0] > 0.999 and th[-1] == 0 and sg[-1] == 0
    ref_times = list(reversed(torch.linspace(-1, T - 1, steps=11).int().tolist()))
    assert bridge_times(T, 10) == list(zip(ref_times[:-1], ref_times[1:]))


def test_bridge_sampler_with_oracle_lands_on_target_along_the_mean_path():
    th, sg = bridge_schedule(50)
    rng = np.random.default_rng(0)
    mu, x0 = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
    x = mu.copy()
    for t, tn in bridge_times(50, 7):
        x = bridge_step(x, mu, x0, th, sg, t, tn)
        if tn >= 0:
            assert np.allclose(x, mu + th[tn] * (x0 - mu))       # noiseless marginal of q(x_t | x0, mu)
    assert np.allclose(x, x0)


@pytest.mark.parametrize("residual_noise", [False, True])
def test_rdbm_train_sample_export(scenes, tmp_path, residual_noise):  # noqa: F811
    """Train a bridge model with the tile_range input (initialised from a Gaussian GrounDiff checkpoint
    without it); torch, numpy and ONNX samplers agree, make bridge_steps calls, and are deterministic."""
    base = tmp_path / "gauss"
    train(base_cfg(scenes, base))
    out = tmp_path / "rdbm"
    chans = ["dsm_max", "dsm_min", "dtm_before", "sem_ground", "tile_range"]
    cfg = base_cfg(scenes, out, diffusion={"process": "rdbm", "bridge_T": 20, "bridge_steps": 4,
                                           "bridge_residual_noise": residual_noise},
                   data={"cond_channels": chans})
    train(cfg, init_from=str(base / "last.pt"))
    model, cfg1, _ = load_model(out / "last.pt")
    assert cfg1.diffusion.process == "rdbm" and model.bridge
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    torch.manual_seed(0)
    cond = torch.randn(2, len(chans), 32, 32)
    a, la = model.sample(cond)
    b, _ = model.sample(cond, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)                                     # deterministic sampler
    from groundiff.backends import TorchNet
    spec = RuntimeSpec.from_config(cfg1)
    assert spec.process == "rdbm" and spec.steps == 4 and spec.deterministic
    g, lg = sample(TorchNet(model, "cpu"), cond.numpy(), spec)
    assert np.allclose(a.numpy(), g, atol=2e-4) and np.allclose(la.numpy(), lg, atol=2e-3)
    # exported ONNX + numpy runtime (as QGIS runs it); weights saved before the perturbation above,
    # so compare against the reloaded checkpoint
    model2, _, _ = load_model(out / "last.pt")
    ref, _ = model2.sample(cond)
    onnx_path, _ = export(out / "last.pt", tmp_path / "m")
    spec2 = RuntimeSpec.from_json(tmp_path / "m.json")
    assert spec2.process == "rdbm" and "tile_range" not in spec2.needed_channels
    net = OnnxNet(onnx_path, "cpu")
    calls = []
    counted = lambda *x: (calls.append(1), net(*x))[1]          # noqa: E731
    g2, _ = sample(counted, cond.numpy(), spec2)
    assert len(calls) == 4 and np.abs(g2 - ref.numpy()).max() < 1e-4
    sd = scenes / "scenes" / "S2"
    arrs = {n: np.load(sd / f"{n}.npy") for n in spec2.needed_channels}
    res = predict_scene(arrs, spec2, net, batch_size=4, n_samples=3)
    assert np.isfinite(res["dtm"]).any() and "std" not in res


def test_tile_range_channel_is_the_same_in_training_and_inference(scenes):  # noqa: F811
    from groundiff.config import config_from_dict
    from groundiff.data.dataset import TileDataset
    from groundiff.runtime import prepare, tile_norm
    cfg = base_cfg(scenes, scenes / "unused", data={"cond_channels": ["dsm_max", "tile_range"], "augment": False})
    ds = TileDataset(cfg.data, split="val", mode="eval")
    item = ds[0]
    tr = item["cond"][1].numpy()
    assert np.ptp(tr) == 0 and np.isclose(tr[0, 0], 0.5 * np.log(float(item["scale"]) / 10.0), atol=1e-6)
    spec = RuntimeSpec.from_config(cfg)
    si, r0, c0, _ = ds.index[0]
    sc = ds.scenes[si]
    arrs = {n: sc.window(n, r0, c0, 32, 32) for n in spec.needed_channels}
    lo, scale = tile_norm(arrs, spec)
    x = prepare(arrs, spec, lo, scale)
    assert np.allclose(x, item["cond"].numpy(), atol=1e-5)

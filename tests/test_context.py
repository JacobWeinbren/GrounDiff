"""Wide-area context channels ("PrioStitch as an input") and the network's context branch."""
import numpy as np
import pytest
import torch

from groundiff.backends import OnnxNet, TorchNet
from groundiff.data.dataset import TileDataset, check_context_order
from groundiff.export import export
from groundiff.models.build import build_model, init_from_checkpoint, load_model
from groundiff.normalise import context_window
from groundiff.runtime import RuntimeSpec, predict_scene, prepare, sample, tile_norm
from groundiff.train import train
from tests.test_train import base_cfg, scenes  # noqa: F401  (fixture)

CTX = ["ctx_dsm_min", "ctx_dsm_max", "ctx_has_return"]
CHANS = ["dsm_max", "dsm_min", "dtm_before", "sem_ground", "tile_range"] + CTX


def test_context_window_pools_the_centred_wider_window():
    a = np.arange(40 * 40, dtype=np.float32).reshape(40, 40)
    a[0, 0] = np.nan
    t, F = 8, 4                                   # tile at (16, 16): context rows/cols 4 .. 35
    mn = context_window(a, 16, 16, t, F, "min")
    mx = context_window(a, 16, 16, t, F, "max")
    me = context_window(a, 16, 16, t, F, "mean")
    blk = a[4:36, 4:36].reshape(t, F, t, F)
    assert np.array_equal(mn, blk.min(axis=(1, 3))) and np.array_equal(mx, blk.max(axis=(1, 3)))
    assert np.allclose(me, blk.mean(axis=(1, 3)))
    edge = context_window(a, 0, 0, t, F, "mean")  # mostly outside: NaN there, NaN-aware inside
    assert np.isnan(edge[0, 0]) and np.isfinite(edge[-1, -1])
    with pytest.raises(ValueError):
        check_context_order(["ctx_dsm_min", "dsm_max"])


def test_training_and_runtime_context_agree(scenes):  # noqa: F811
    cfg = base_cfg(scenes, scenes / "unused", data={"cond_channels": CHANS, "augment": False})
    ds = TileDataset(cfg.data, split="val", mode="eval")
    spec = RuntimeSpec.from_config(cfg)
    assert spec.context_margin == 3 * 32 // 2 and "ctx_dsm_min" not in spec.needed_channels
    for i in range(min(4, len(ds))):
        item = ds[i]
        si, r0, c0, _ = ds.index[i]
        sc = ds.scenes[si]
        full = {n: sc.array(n) for n in spec.needed_channels}
        ta = spec.add_context({n: sc.window(n, r0, c0, 32, 32) for n in spec.needed_channels}, full, r0, c0)
        lo, scale = tile_norm(ta, spec)
        assert np.allclose(prepare(ta, spec, lo, scale), item["cond"].numpy(), atol=1e-5)


def test_augmented_context_is_finite_and_follows_the_tile(scenes):  # noqa: F811
    cfg = base_cfg(scenes, scenes / "unused", data={"cond_channels": CHANS, "p_jitter": 1.0,
                                                   "p_multiscale": 1.0})
    ds = TileDataset(cfg.data, split="train", mode="train")
    ds.set_epoch(0)
    for i in range(6):
        it = ds[i]
        assert it["cond"].shape[0] == len(CHANS) and torch.isfinite(it["cond"]).all()


def test_context_model_finetune_train_and_export(scenes, tmp_path):  # noqa: F811
    base = tmp_path / "base"
    train(base_cfg(scenes, base))
    cfg = base_cfg(scenes, tmp_path / "ctx", data={"cond_channels": CHANS})
    # fine-tune start: the new model reproduces the old one exactly (zero-initialised context projection)
    old, old_cfg, _ = load_model(base / "last.pt")
    with torch.no_grad():
        for p in old.parameters():
            p.add_(0.02 * torch.randn_like(p))
    torch.save({"config": old_cfg.to_dict(), "model": old.state_dict()}, tmp_path / "perturbed.pt")
    new = build_model(cfg)
    notes = init_from_checkpoint(new, cfg, tmp_path / "perturbed.pt")
    assert any("tile_range" in n for n in notes)
    new.eval()
    torch.manual_seed(1)
    cond_new = torch.randn(2, len(CHANS), 32, 32)
    cond_old = cond_new[:, [CHANS.index(c) for c in old_cfg.data.cond_channels]]
    g_t, gamma = torch.randn(2, 1, 32, 32), torch.rand(2)
    with torch.no_grad():
        a = old.denoise(g_t, cond_old, gamma)[0]
        b = new.denoise(g_t, cond_new, gamma)[0]
    assert torch.allclose(a, b, atol=1e-5)
    # train, then torch / numpy / ONNX agree and predict_scene reads the context around each tile
    out = tmp_path / "ctx"
    train(cfg, init_from=str(base / "last.pt"))
    model, cfg1, _ = load_model(out / "last.pt")
    spec = RuntimeSpec.from_config(cfg1)
    x, _ = model.sample(cond_new, init="dsm", add_noise=False)
    y, _ = sample(TorchNet(model, "cpu"), cond_new.numpy(), spec, init="dsm", add_noise=False)
    assert np.allclose(x.numpy(), y, atol=2e-4)
    onnx_path, _ = export(out / "last.pt", tmp_path / "m")
    spec2 = RuntimeSpec.from_json(tmp_path / "m.json")
    assert spec2.context_channels == CTX and spec2.context_factor == 4
    net = OnnxNet(onnx_path, "cpu")
    z, _ = sample(net, cond_new.numpy(), spec2, init="dsm", add_noise=False)
    assert np.abs(z - x.numpy()).max() < 1e-4
    sd = scenes / "scenes" / "S2"
    arrs = {n: np.load(sd / f"{n}.npy") for n in spec2.needed_channels}
    res = predict_scene(arrs, spec2, net, batch_size=4, n_samples=1)
    assert np.isfinite(res["dtm"]).any()

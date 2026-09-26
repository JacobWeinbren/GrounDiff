import json

import pytest
import torch

from groundiff.config import config_from_dict
from groundiff.data.preprocess import process_scene
from groundiff.models.build import build_model, init_from_checkpoint, load_model
from groundiff.train import train
from tests.synthetic import write_scene

TINY_MODEL = {"inner_channel": 32, "channel_mults": [1, 2], "num_head_channels": 32, "dropout": 0.0}


@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    root = tmp_path_factory.mktemp("train")
    for i in range(3):
        after, before = write_scene(root / "laz", name=f"S{i}", seed=i, size=96.0)
        process_scene(after, root / "scenes", before=before, gsd=1.0)
    split = {"train": ["S0", "S1"], "val": ["S2"], "test": []}
    (root / "split.json").write_text(json.dumps(split))
    return root


def base_cfg(root, out, **over):
    d = {
        "data": {"root": str(root / "scenes"), "split_file": str(root / "split.json"), "tile": 32,
                 "samples_per_epoch": 8, "cond_channels": ["dsm_max", "dsm_min", "dtm_before", "sem_ground"],
                 "norm_channels": ["dsm_max", "dsm_min", "dtm_before"], "prior_channel": "dtm_before",
                 "multiscale_sizes": [32, 64, 128]},
        "model": dict(TINY_MODEL),
        "optim": {"total_steps": 6, "warmup_steps": 2, "lr": 1e-3},
        "train": {"out_dir": str(out), "batch_size": 2, "grad_accum": 2, "num_workers": 0,
                  "log_every": 2, "val_every": 3, "save_every": 3, "val_max_tiles": 4, "device": "cpu"},
    }
    for sec, vals in over.items():
        d.setdefault(sec, {}).update(vals)
    return config_from_dict(d)


def test_groundiff_train_resume_and_reload(scenes, tmp_path):
    out = tmp_path / "gd"
    best = train(base_cfg(scenes, out))
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    assert best["rmse"] < float("inf")
    events = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    vals = [e for e in events if e["event"] == "val"]
    assert vals and "lasground_new" in vals[0]["raw"] and "ema" in vals[0]
    # resume: extend the run by two steps
    train(base_cfg(scenes, out, optim={"total_steps": 8, "warmup_steps": 2, "lr": 1e-3}))
    ck = torch.load(out / "last.pt", weights_only=False)
    assert ck["step"] == 8
    model, cfg, _ = load_model(out / "best.pt")
    assert cfg.data.cond_channels[2] == "dtm_before"


def test_init_from_widens_input_exactly(scenes, tmp_path):
    out = tmp_path / "small"
    train(base_cfg(scenes, out, data={"cond_channels": ["dsm_max", "dsm_min"],
                                      "norm_channels": ["dsm_max", "dsm_min"], "prior_channel": None},
                   optim={"total_steps": 3, "warmup_steps": 1, "lr": 1e-3}))
    old, old_cfg, _ = load_model(out / "last.pt", use_ema=True)   # init_from uses the EMA weights
    new_cfg = base_cfg(scenes, tmp_path / "wide")
    new = build_model(new_cfg)
    notes = init_from_checkpoint(new, new_cfg, out / "last.pt")
    assert any("stem widened" in n for n in notes)
    new.eval()
    torch.manual_seed(0)
    g_t, cond_old = torch.randn(1, 1, 32, 32), torch.randn(1, 2, 32, 32)
    extra = torch.randn(1, 2, 32, 32)
    cond_new = torch.cat([cond_old, extra], 1)          # dsm_max, dsm_min, dtm_before, sem_ground
    gamma = torch.tensor([0.5])
    with torch.no_grad():
        a = old.denoiser(torch.cat([g_t, cond_old], 1), gamma)
        b = new.denoiser(torch.cat([g_t, cond_new], 1), gamma)
    assert torch.allclose(a, b, atol=1e-5)


def test_resdepth_trains(scenes, tmp_path):
    out = tmp_path / "rd"
    cfg = base_cfg(scenes, out,
                   data={"norm_mode": "mean_std", "norm_channels": ["dtm_before"], "tile": 32},
                   model={"kind": "resdepth", "resdepth_depth": 3, "resdepth_start_kernel": 16},
                   optim={"name": "adam", "schedule": "step", "step_size": 100, "total_steps": 4,
                          "lr": 2e-4, "weight_decay": 0.0, "ema_decay": None})
    best = train(cfg)
    assert best["rmse"] < float("inf")
    saved = json.loads((out / "config.json").read_text())
    assert saved["data"]["norm_std"] and saved["data"]["norm_std"] > 0


def test_validation_restores_live_weights():
    from groundiff.train import EMA, load_plain_state
    m = build_model(config_from_dict({"model": TINY_MODEL, "data": {"cond_channels": ["dsm_max"],
                                                                    "norm_channels": ["dsm_max"]}}))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    ema = EMA(m, 0.9)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    live = {k: v.clone() for k, v in m.state_dict().items()}
    backup = load_plain_state(m, ema.state_dict(m))
    k = next(k for k in before if k.endswith("weight"))
    assert torch.equal(m.state_dict()[k], before[k])          # EMA weights swapped in
    m.load_state_dict(backup)
    assert all(torch.equal(m.state_dict()[k], live[k]) for k in live)   # and restored


def test_before_after_gate_on_lasground(scenes, tmp_path):
    from groundiff.runtime import RuntimeSpec, predict_scene
    from groundiff.backends import TorchNet
    import numpy as np
    out = tmp_path / "ba"
    cfg = base_cfg(scenes, out, data={"gate_channel": "dtm_before", "fill_empty": "nearest", "loss_mask": "gt"},
                   loss={"units": "metres"}, optim={"total_steps": 3, "warmup_steps": 1, "lr": 1e-3})
    train(cfg)
    model, cfg2, _ = load_model(out / "last.pt")
    spec = RuntimeSpec.from_config(cfg2)
    assert spec.fill_empty == "nearest" and spec.gate_channel == "dtm_before"
    sd = scenes / "scenes" / "S2"
    arrs = {n: np.load(sd / f"{n}.npy") for n in spec.needed_channels}
    res = predict_scene(arrs, spec, TorchNet(model, "cpu"), batch_size=4)
    assert "p_edit" in res and np.allclose(res["p_edit"] + res["p_ground"], 1.0, equal_nan=True)


def test_monitor_reads_a_run(scenes, tmp_path):
    from groundiff.monitor import summary, to_csv
    out = tmp_path / "mon"
    train(base_cfg(scenes, out, optim={"total_steps": 3, "warmup_steps": 1, "lr": 1e-3}))
    s = summary(out)
    assert "step 3/3" in s and "validation @ step 3" in s and "lasground_new RMSE" in s and "DONE" in s
    to_csv(out, tmp_path / "c.csv")
    assert (tmp_path / "c.csv").read_text().startswith("conf") or "loss" in (tmp_path / "c.csv").read_text()

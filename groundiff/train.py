"""Training (GrounDiff or ResDepth) on CUDA, Apple MPS or CPU.

    python -m groundiff.train configs/before_after.json
    python -m groundiff.train configs/before_after.json --set train.batch_size=4 train.grad_accum=4

Resumes automatically from <out_dir>/last.pt. Writes:
    last.pt         full state (model, EMA, optimiser, scheduler, step)
    best.pt         weights (raw + EMA) with the best validation RMSE
    log.jsonl       one JSON line per log / validation event
    config.json     the resolved config
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config, load_config
from .data.dataset import TileDataset
from .device import autocast, memory_gb, pick_device, resolve_amp, setup_backend
from .diffusion import GrounDiff
from .evaluation import evaluate_loader
from .losses import groundiff_loss, resdepth_loss
from .models.build import build_model, cpu_state, init_from_checkpoint, save_checkpoint


def _plain(k: str) -> str:
    return k.replace("_orig_mod.", "")


def is_diff_model(model) -> bool:
    return isinstance(model, GrounDiff)


class EMA:
    """Exponential moving average of floating-point weights (keys are stored
    without torch.compile's `_orig_mod.` prefix)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {_plain(k): v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int):
        d = min(self.decay, (1 + step) / (10 + step))      # short warm-up of the average
        for k, v in model.state_dict().items():
            k = _plain(k)
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    def state_dict(self, model: torch.nn.Module) -> dict:
        sd = cpu_state(model)
        sd.update({k: v.detach().cpu() for k, v in self.shadow.items()})
        return sd

    def load(self, sd: dict):
        for k in self.shadow:
            if k in sd:
                self.shadow[k].copy_(sd[k])


def load_plain_state(model: torch.nn.Module, sd: dict) -> dict:
    """Load a prefix-free state dict into a (possibly compiled) model and
    return the previous live state for restoring."""
    live = model.state_dict()
    backup = {k: v.detach().clone() for k, v in live.items()}
    model.load_state_dict({k: sd[_plain(k)] for k in live})
    return backup


def lr_lambda(cfg: Config):
    o = cfg.optim
    if o.schedule == "step":
        return lambda s: o.step_gamma ** (s // max(o.step_size, 1))
    if o.schedule == "constant":
        return lambda s: 1.0

    def f(s):
        if s < o.warmup_steps:
            return (s + 1) / max(o.warmup_steps, 1)
        prog = min(1.0, (s - o.warmup_steps) / max(o.total_steps - o.warmup_steps, 1))
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return (o.min_lr + (o.lr - o.min_lr) * cos) / o.lr
    return f


def make_optimizer(cfg: Config, model: torch.nn.Module):
    o = cfg.optim
    params = [p for p in model.parameters() if p.requires_grad]
    if o.name == "adamw":
        return torch.optim.AdamW(params, lr=o.lr, betas=tuple(o.betas), weight_decay=o.weight_decay)
    if o.name == "adam":
        return torch.optim.Adam(params, lr=o.lr, betas=tuple(o.betas), weight_decay=o.weight_decay)
    raise ValueError(o.name)


def estimate_norm_std(ds: TileDataset, n: int = 200) -> float:
    """ResDepth (train.py, compute_local_dsm_std_per_centered_patch): std of
    the initial raster after centring each patch on its mean."""
    saved = (ds.cfg.norm_mode, ds.cfg.norm_std)
    ds.cfg.norm_mode, ds.cfg.norm_std = "mean_std", 1.0
    vals = []
    for i in range(min(n, len(ds))):
        it = ds[i]
        v = it["valid"][0] > 0
        vals.append(it["prior"][0][v].numpy())         # already mean-centred (std = 1)
    ds.cfg.norm_mode, ds.cfg.norm_std = saved
    return float(np.concatenate(vals).std()) if vals else 1.0


def infinite_batches(ds: TileDataset, cfg: Config, device: torch.device):
    epoch = 0
    while True:
        ds.set_epoch(epoch)
        dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False, drop_last=True,
                        num_workers=cfg.train.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=False)
        for b in dl:
            yield b
        epoch += 1


def log_event(path: Path, event: dict):
    line = json.dumps(event, default=float)
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def train(cfg: Config, init_from: str | None = None) -> dict:
    out = Path(cfg.train.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.train.device)
    setup_backend(device)
    amp_dtype = resolve_amp(cfg.train.amp, device)
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    train_ds = TileDataset(cfg.data, "train" if cfg.data.split_file else None, mode="train")
    val_ds = TileDataset(cfg.data, "val" if cfg.data.split_file else None, mode="eval")
    if cfg.data.norm_mode == "mean_std" and cfg.data.norm_std is None:
        cfg.data.norm_std = estimate_norm_std(train_ds)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = make_optimizer(cfg, model)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg))
    ema = EMA(model, cfg.optim.ema_decay) if cfg.optim.ema_decay else None
    scaler = torch.amp.GradScaler(device.type) if amp_dtype == torch.float16 else None

    step, best = 0, {"rmse": float("inf")}
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        if ema and ck.get("ema"):
            ema.load(ck["ema"])
        step, best = ck["step"], ck.get("best", best)
        print(f"resumed from {last} at step {step}")
    elif init_from:
        for note in init_from_checkpoint(model, cfg, init_from):
            print("init:", note)
        if ema:
            ema = EMA(model, cfg.optim.ema_decay)

    if cfg.train.compile:
        if is_diff_model(model):
            model.denoiser = torch.compile(model.denoiser)
        else:
            model.net = torch.compile(model.net)
    log_path = out / "log.jsonl"
    log_event(log_path, {"event": "start", "device": str(device), "amp": str(amp_dtype),
                         "params_M": n_params / 1e6, "train_tiles_per_epoch": len(train_ds),
                         "val_tiles": len(val_ds), "step": step})
    batches = infinite_batches(train_ds, cfg, device)
    is_diff = is_diff_model(model)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers)

    def validate(tag_step: int):
        results = {}
        for name, weights in (("raw", None), ("ema", ema.state_dict(model) if ema else None)):
            if name == "ema" and weights is None:
                continue
            backup = load_plain_state(model, weights) if weights is not None else None
            results[name] = evaluate_loader(model, val_loader, cfg, device, cfg.train.val_max_tiles,
                                            init=cfg.train.val_init, t_start=cfg.train.val_t_start)
            if backup is not None:
                model.load_state_dict(backup)
        model.train()
        return results

    model.train()
    t0, acc, n_acc = time.time(), {}, 0
    while step < cfg.optim.total_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.train.grad_accum):
            b = next(batches)
            cond, target = b["cond"].to(device, non_blocking=True), b["target"].to(device, non_blocking=True)
            valid = b["valid"].to(device, non_blocking=True)
            with autocast(device, amp_dtype):
                if is_diff:
                    o = model.training_forward(target, cond)
                    losses = groundiff_loss(o["g0_hat"], o["logit"], target, b["m_alpha"].to(device), valid, cfg.loss)
                else:
                    pred = model(b["prior"].to(device), cond)
                    losses = resdepth_loss(pred.float(), target, valid, 0.5 * b["scale"].to(device))
            loss = losses["loss"] / cfg.train.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {losses}")
            (scaler.scale(loss) if scaler else loss).backward()
            for k, v in losses.items():
                acc[k] = acc.get(k, 0.0) + float(v.detach()) / cfg.train.grad_accum
        n_acc += 1
        if scaler:
            scaler.unscale_(opt)
        if cfg.optim.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        sched.step()
        step += 1
        if ema:
            ema.update(model, step)

        if step % cfg.train.log_every == 0 or step == 1:
            dt = time.time() - t0
            log_event(log_path, {"event": "train", "step": step, "lr": sched.get_last_lr()[0],
                                 **{k: v / n_acc for k, v in acc.items()},
                                 "sec_per_step": dt / n_acc, **memory_gb(device)})
            t0, acc, n_acc = time.time(), {}, 0
        if step % cfg.train.val_every == 0 or step == cfg.optim.total_steps:
            res = validate(step)
            log_event(log_path, {"event": "val", "step": step, **res})
            for name, r in res.items():
                rmse = r["model"].get("rmse", float("inf"))
                if rmse < best["rmse"]:
                    best = {"rmse": rmse, "step": step, "weights": name, "metrics": r}
                    sd = cpu_state(model) if name == "raw" else ema.state_dict(model)
                    save_checkpoint(out / "best.pt", cfg, sd, step=step, weights=name, metrics=r)
        if step % cfg.train.save_every == 0 or step == cfg.optim.total_steps:
            save_checkpoint(last, cfg, cpu_state(model), ema=ema.state_dict(model) if ema else None,
                            optimizer=opt.state_dict(), scheduler=sched.state_dict(), step=step, best=best)
    log_event(log_path, {"event": "done", "step": step, "best": {k: v for k, v in best.items() if k != "metrics"}})
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.batch_size=4")
    ap.add_argument("--init-from", help="checkpoint to fine-tune from (new input channels zero-initialised)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    train(cfg, a.init_from)
    return 0


if __name__ == "__main__":
    sys.exit(main())

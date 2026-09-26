"""Model construction and portable checkpoints.

Checkpoints hold CPU tensors plus the full config, so a model trained on a Mac
(MPS) loads unchanged on a CUDA PC or CPU.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..config import Config, config_from_dict
from ..diffusion import GrounDiff
from .resdepth import build_resdepth
from .unet import UNet


class ResDepthModel(nn.Module):
    """ResDepth: channel 0 of the input is the initial DTM (the prior
    channel); the remaining conditioning channels are guidance."""

    def __init__(self, cfg: Config):
        super().__init__()
        if not cfg.data.prior_channel:
            raise ValueError("ResDepth needs data.prior_channel (e.g. 'dtm_before')")
        self.cond_channels = [c for c in cfg.data.cond_channels if c != cfg.data.prior_channel]
        self.net = build_resdepth(1 + len(self.cond_channels), cfg.model.resdepth_depth,
                                  cfg.model.resdepth_start_kernel)
        self._keep = [cfg.data.cond_channels.index(c) for c in self.cond_channels]

    def forward(self, prior: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([prior, cond[:, self._keep]], dim=1))


def build_model(cfg: Config) -> nn.Module:
    m = cfg.model
    if m.kind == "groundiff":
        net = UNet(in_channel=1 + len(cfg.data.cond_channels), out_channel=2,
                   inner_channel=m.inner_channel, channel_mults=tuple(m.channel_mults),
                   res_blocks=m.res_blocks, attn_res=tuple(m.attn_res), dropout=m.dropout,
                   num_head_channels=m.num_head_channels, use_checkpoint=m.use_checkpoint)
        return GrounDiff(net, cfg.diffusion, cfg.data.cond_channels, cfg.data.gate_channel)
    if m.kind == "resdepth":
        return ResDepthModel(cfg)
    raise ValueError(f"unknown model kind {m.kind!r}")


def cpu_state(module: nn.Module) -> dict:
    sd = module.state_dict()
    return {k.replace("_orig_mod.", ""): v.detach().to("cpu") for k, v in sd.items()}


def save_checkpoint(path: str | Path, cfg: Config, model_sd: dict, **extra):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"config": cfg.to_dict(), "model": model_sd, **extra}, tmp)
    tmp.replace(path)                     # atomic: never leaves a half-written checkpoint


def load_model(path: str | Path, device: str | torch.device = "cpu", use_ema: bool = True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ck["config"])
    model = build_model(cfg)
    sd = ck.get("ema") if use_ema and ck.get("ema") is not None else ck["model"]
    model.load_state_dict(sd)
    return model.to(device).eval(), cfg, ck


def init_from_checkpoint(model: nn.Module, cfg: Config, path: str | Path) -> list[str]:
    """Fine-tune start: copy matching weights from another checkpoint.

    If the new model has extra conditioning channels (e.g. dtm_before and
    sem_* added to a DSM-only model), the first convolution's weights for
    channels the old model had are copied by channel name and the new ones
    are zero, so the fine-tuned model starts out computing exactly what the
    old one did."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    old_cfg = config_from_dict(ck["config"])
    src = ck.get("ema") or ck["model"]
    dst = model.state_dict()
    notes = []
    for k, v in dst.items():
        if k not in src:
            notes.append(f"new: {k}")
            continue
        w = src[k]
        if w.shape == v.shape:
            dst[k] = w
        elif k.endswith("input_blocks.0.0.weight") and w.shape[0] == v.shape[0]:
            old_names = ["g_t"] + list(old_cfg.data.cond_channels)
            new_names = ["g_t"] + list(cfg.data.cond_channels)
            nw = torch.zeros_like(v)
            for j, name in enumerate(new_names):
                if name in old_names:
                    nw[:, j] = w[:, old_names.index(name)]
            dst[k] = nw
            notes.append(f"stem widened {tuple(w.shape)} -> {tuple(v.shape)}; new channels zero-initialised")
        else:
            notes.append(f"shape mismatch, left at init: {k} {tuple(w.shape)} vs {tuple(v.shape)}")
    model.load_state_dict(dst)
    return notes

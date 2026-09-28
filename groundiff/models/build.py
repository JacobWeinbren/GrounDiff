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
from ..normalise import is_context
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
        from ..data.dataset import check_context_order
        check_context_order(cfg.data.cond_channels)
        n_ctx = sum(is_context(c) for c in cfg.data.cond_channels)
        net = UNet(in_channel=1 + len(cfg.data.cond_channels), out_channel=3 if m.aleatoric else 2,
                   inner_channel=m.inner_channel, channel_mults=tuple(m.channel_mults),
                   res_blocks=m.res_blocks, attn_res=tuple(m.attn_res), dropout=m.dropout,
                   num_head_channels=m.num_head_channels, use_checkpoint=m.use_checkpoint,
                   context_channels=n_ctx, context_factor=cfg.data.context_factor)
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

    GrounDiff models: the first convolution's input weights are always mapped by channel name
    (g_t plus the conditioning channels), so reordered, added or dropped
    channels line up; channels the old model lacked start at zero. The new
    model reproduces the old one exactly only if the gate channel,
    normalisation and no-data filling are also unchanged; the returned notes
    say when they are not (e.g. paper_dsm2dtm -> before_after changes the
    gate from dsm_max to dtm_before, so it starts from the old features, not
    the old output). ResDepth models: weights are copied where shapes match
    (by position; no channel-name mapping)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    old_cfg = config_from_dict(ck["config"])
    src = ck.get("ema") or ck["model"]
    dst = model.state_dict()
    # the stem sees the main channels; context channels go to the context branch
    old_names = ["g_t"] + [c for c in old_cfg.data.cond_channels if not is_context(c)]
    new_names = ["g_t"] + [c for c in cfg.data.cond_channels if not is_context(c)]
    notes = []
    for k, v in dst.items():
        if k not in src:
            notes.append(f"new: {k}")
            continue
        w = src[k]
        stem = k.endswith("input_blocks.0.0.weight") and cfg.model.kind == "groundiff"
        if stem and w.shape[0] == v.shape[0] and (old_names != new_names or w.shape != v.shape):
            nw = torch.zeros_like(v)
            for j, name in enumerate(new_names):
                if name in old_names:
                    nw[:, j] = w[:, old_names.index(name)]
            dst[k] = nw
            added = [n for n in new_names if n not in old_names]
            dropped = [n for n in old_names if n not in new_names]
            notes.append(f"stem mapped by channel name {tuple(w.shape)} -> {tuple(v.shape)}"
                         + (f"; zero-initialised: {added}" if added else "")
                         + (f"; dropped: {dropped}" if dropped else ""))
        elif w.shape == v.shape:
            dst[k] = w
        elif k.endswith(("out.2.weight", "out.2.bias")) and w.shape[1:] == v.shape[1:]:
            n = min(w.shape[0], v.shape[0])              # output head gained / lost the noise-scale channel
            nw = v.clone()
            nw[:n] = w[:n]
            dst[k] = nw
            notes.append(f"output head {tuple(w.shape)} -> {tuple(v.shape)}: first {n} channels copied")
        else:
            notes.append(f"shape mismatch, left at init: {k} {tuple(w.shape)} vs {tuple(v.shape)}")
    od, nd = old_cfg.data, cfg.data
    for attr in ("gate_channel", "norm_channels", "norm_mode", "fill_empty", "prior_channel"):
        if getattr(od, attr) != getattr(nd, attr):
            notes.append(f"{attr} changed ({getattr(od, attr)!r} -> {getattr(nd, attr)!r}): "
                         "the fine-tune starts from the old features, not the old output")
    model.load_state_dict(dst)
    return notes

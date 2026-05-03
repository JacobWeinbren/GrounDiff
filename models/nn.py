"""Small NN utilities used across GrounDiff modules.

Adapted from openai/guided-diffusion via Palette
(Janspiry/Palette-Image-to-Image-Diffusion-Models). Kept intentionally
minimal: GroupNorm, SiLU shim, gamma sinusoidal embedding, zero_module
helper.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class GroupNorm32(nn.GroupNorm):
    """GroupNorm that always computes in float32 then casts back.
    Stabler under fp16 training."""

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int) -> nn.Module:
    """32-group GroupNorm (or fewer if channels < 32)."""
    g = 32
    while g > 1 and channels % g != 0:
        g //= 2
    return GroupNorm32(g, channels)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module — used on the final out-conv
    so the residual prediction starts at 0 (helps training stability)."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def gamma_embedding(gammas: torch.Tensor, dim: int,
                    max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding of γ (cumulative noise level).

    Palette's convention: encode γ rather than t directly. γ_t = ᾱ_t,
    so values lie in (0, 1] across the schedule. Same sin/cos form as
    DDPM but applied to γ instead of integer t.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=gammas.device)
        / half
    )
    args = gammas[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def kaiming_init_(module: nn.Module) -> int:
    """Apply Kaiming-normal init to all Conv*/Linear weights, bias=0.

    Mirrors Palette `core/base_network.py::init_weights` with
    init_type='kaiming': nn.init.kaiming_normal_(weight, a=0, mode='fan_in').

    Skips modules whose weights have already been zero-init'd (the final
    output conv and ResBlock out_layers — `zero_module` made them all-zero
    on purpose so the residual starts at 0). We detect those by checking
    whether the L1 norm of the weight is exactly zero before re-init.

    Returns: number of modules initialised.
    """
    count = 0
    for m in module.modules():
        cls_name = m.__class__.__name__
        if 'Conv' not in cls_name and 'Linear' not in cls_name:
            continue
        if not hasattr(m, 'weight') or m.weight is None:
            continue
        # Skip zero-init'd modules (preserve `zero_module` semantics)
        with torch.no_grad():
            if float(m.weight.detach().abs().sum().item()) == 0.0:
                continue
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
        count += 1
    return count

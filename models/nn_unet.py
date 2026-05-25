"""Small NN utilities for the UNet backbone.

Adapted from openai/guided-diffusion via Palette
(Janspiry/Palette-Image-to-Image-Diffusion-Models). Kept intentionally
minimal: GroupNorm wrapper, gamma sinusoidal embedding, zero_module
helper. These are shared by `unet.py`.

The DiT backbone has its own analogues (sinusoidal embedder in
`patch_embed.py`, no GroupNorm) so we keep them separate.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class GroupNorm32(nn.GroupNorm):
    """GroupNorm that computes in fp32 then casts back to the input dtype.

    Matches openai/guided-diffusion and the official GrounDiff
    `clean rebuild`. With bf16 autocast active, GroupNorm's mean/var
    reductions sum over tens of thousands of activations per group;
    keeping the reduction in fp32 avoids accumulation error that can
    destabilise training. The cost is one extra fp16↔fp32 roundtrip per
    norm — small relative to the convs around it.
    """

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int) -> nn.Module:
    """32-group GroupNorm (or fewer if channels < 32). The 32-group
    pattern is the OpenAI guided-diffusion default."""
    g = 32
    while g > 1 and channels % g != 0:
        g //= 2
    return GroupNorm32(g, channels)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module — used on the final out-conv
    so the predicted residual starts at exactly 0 (DiT-style identity-at-
    init for UNet, also helps training stability)."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def gamma_embedding(gammas: torch.Tensor, dim: int,
                    max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding of γ (cumulative noise level).

    Palette / GrounDiff convention: encode γ rather than integer t. With
    T=10 our γ values are about
        γ ∈ {0.92, 0.83, 0.74, ..., 0.0034}
    spanning roughly (0, 1]. The sinusoidal embedding is then the standard
    DDPM positional-style encoding applied to γ.
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

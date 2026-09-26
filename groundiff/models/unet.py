"""Denoiser U-Net: a port of Palette's guided-diffusion U-Net.

Source: janspiry/palette-image-to-image-diffusion-models
(models/guided_diffusion_modules/{unet,nn}.py), MIT licence,
Copyright (c) 2022 Liangwei Jiang, itself derived from openai/guided-diffusion
(MIT). The GrounDiff first author recommended Palette as the baseline, and
Palette's default configuration (inner_channel=64, channel_mults=(1,2,4,8),
res_blocks=2, attention only in the middle block, 32 channels per head)
has 62.64M parameters with 2 output channels, which matches the 62.6M reported
in GrounDiff §8.1.

Changes from Palette, none of which alter the maths:
  * gradient checkpointing uses torch.utils.checkpoint (works on CUDA, MPS
    and CPU) and is off unless `use_checkpoint=True`;
  * attention uses F.scaled_dot_product_attention (same result as the
    legacy einsum path, less memory);
  * mixed precision comes from autocast in the training loop; GroupNorm, the
    input convolution and the output head always run in float32, so bf16 does
    not quantise input heights or the predicted residual (bf16 keeps ~3
    significant digits: ~8 cm steps on a 20 m-relief tile).
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int) -> nn.Module:
    return GroupNorm32(32, channels)


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


def gamma_embedding(gammas: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of the continuous noise level (Palette)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=gammas.device) / half)
    args = gammas[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class EmbedSequential(nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, (ResBlock, EmbedSequential)) else layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, out_channel: int | None = None):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, self.out_channel, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, out_channel: int | None = None):
        super().__init__()
        self.out_channel = out_channel or channels
        if use_conv:
            self.op = nn.Conv2d(channels, self.out_channel, 3, stride=2, padding=1)
        else:
            assert channels == self.out_channel
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


class ResBlock(nn.Module):
    """Residual block with FiLM (scale-shift) conditioning on the noise level."""

    def __init__(self, channels, emb_channels, dropout, out_channel=None, use_conv=False,
                 use_scale_shift_norm=True, use_checkpoint=False, up=False, down=False):
        super().__init__()
        self.out_channel = out_channel or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv2d(channels, self.out_channel, 3, padding=1))
        self.updown = up or down
        if up:
            self.h_upd, self.x_upd = Upsample(channels, False), Upsample(channels, False)
        elif down:
            self.h_upd, self.x_upd = Downsample(channels, False), Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channel if use_scale_shift_norm else self.out_channel))
        # out_layers[2] is the dropout. Under activation checkpointing the mask
        # is drawn outside the checkpointed function and passed in, so the
        # recomputation uses the same mask on every device (torch does not save
        # and restore the MPS RNG around checkpoints).
        self.out_layers = nn.Sequential(
            normalization(self.out_channel), nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)))
        self.dropout = float(dropout)
        if self.out_channel == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 1)

    def forward(self, x, emb):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            mask = None
            if self.dropout > 0:
                b, _, hh, ww = x.shape
                if self.updown:
                    hh, ww = (hh * 2, ww * 2) if isinstance(self.h_upd, Upsample) else (hh // 2, ww // 2)
                # boolean: a quarter of the memory of a float mask (kept alive for the recomputation)
                mask = torch.rand(b, self.out_channel, hh, ww, device=x.device) < (1.0 - self.dropout)
            return checkpoint(self._forward, x, emb, mask, use_reentrant=False, preserve_rng_state=False)
        return self._forward(x, emb)

    def _forward(self, x, emb, mask=None):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_conv(self.h_upd(in_rest(x)))
            x = self.x_upd(x)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)[..., None, None]
        norm, act, drop, conv = self.out_layers
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = act(norm(h) * (1 + scale) + shift)
        else:
            h = act(norm(h + emb_out))
        h = h * (mask.to(h.dtype) / (1.0 - self.dropout)) if mask is not None else drop(h)
        return self.skip_connection(x) + conv(h)


class AttentionBlock(nn.Module):
    """Spatial self-attention; heads are split before q/k/v as in Palette's
    default (QKVAttentionLegacy), so parameter layout matches Palette."""

    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False):
        super().__init__()
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        length = qkv.shape[-1]
        ch = c // self.num_heads
        q, k, v = qkv.reshape(b * self.num_heads, ch * 3, length).split(ch, dim=1)
        # SDPA expects (batch, heads, tokens, dim); scale 1/sqrt(ch) equals
        # Palette's (q/ch^0.25)·(k/ch^0.25).
        a = F.scaled_dot_product_attention(
            q.transpose(1, 2)[:, None], k.transpose(1, 2)[:, None], v.transpose(1, 2)[:, None])
        h = a[:, 0].transpose(1, 2).reshape(b, c, length)
        return (x + self.proj_out(h)).reshape(b, c, *spatial)


class UNet(nn.Module):
    def __init__(self, in_channel: int, out_channel: int = 2, inner_channel: int = 64,
                 channel_mults=(1, 2, 4, 8), res_blocks: int = 2, attn_res=(16,),
                 dropout: float = 0.2, num_heads: int = 1, num_head_channels: int = 32,
                 use_scale_shift_norm: bool = True, resblock_updown: bool = True,
                 conv_resample: bool = True, use_checkpoint: bool = False):
        super().__init__()
        self.in_channel = in_channel
        self.inner_channel = inner_channel
        emb_dim = inner_channel * 4
        self.cond_embed = nn.Sequential(
            nn.Linear(inner_channel, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        def res(ch_in, ch_out=None, **kw):
            return ResBlock(ch_in, emb_dim, dropout, out_channel=ch_out,
                            use_scale_shift_norm=use_scale_shift_norm,
                            use_checkpoint=use_checkpoint, **kw)

        def attn(ch):
            return AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels,
                                  use_checkpoint=use_checkpoint)

        ch = input_ch = int(channel_mults[0] * inner_channel)
        self.input_blocks = nn.ModuleList([EmbedSequential(nn.Conv2d(in_channel, ch, 3, padding=1))])
        chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mults):
            for _ in range(res_blocks):
                layers = [res(ch, int(mult * inner_channel))]
                ch = int(mult * inner_channel)
                if ds in attn_res:
                    layers.append(attn(ch))
                self.input_blocks.append(EmbedSequential(*layers))
                chans.append(ch)
            if level != len(channel_mults) - 1:
                self.input_blocks.append(EmbedSequential(
                    res(ch, ch, down=True) if resblock_updown else Downsample(ch, conv_resample)))
                chans.append(ch)
                ds *= 2
        self.middle_block = EmbedSequential(res(ch), attn(ch), res(ch))
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(res_blocks + 1):
                layers = [res(ch + chans.pop(), int(inner_channel * mult))]
                ch = int(inner_channel * mult)
                if ds in attn_res:
                    layers.append(attn(ch))
                if level and i == res_blocks:
                    layers.append(res(ch, ch, up=True) if resblock_updown
                                  else Upsample(ch, conv_resample))
                    ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
        self.out = nn.Sequential(
            normalization(ch), nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channel, 3, padding=1)))

    def forward(self, x: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        emb = self.cond_embed(gamma_embedding(gammas.reshape(-1), self.inner_channel))
        with _fp32(x):
            h = self.input_blocks[0](x.float(), emb)
        hs = [h]
        for module in self.input_blocks[1:]:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), emb)
        with _fp32(x):
            return self.out(h.float())


def _fp32(x: torch.Tensor):
    """Disable autocast locally (no-op when autocast is off)."""
    try:
        if torch.is_autocast_enabled(x.device.type):
            return torch.autocast(device_type=x.device.type, enabled=False)
    except TypeError:                     # torch < 2.4: no device argument
        on = torch.is_autocast_cpu_enabled() if x.device.type == "cpu" else torch.is_autocast_enabled()
        if on:
            return torch.autocast(device_type=x.device.type, enabled=False)
    return contextlib.nullcontext()

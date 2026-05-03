"""GrounDiff U-Net (paper §3.2 + Fig.4).

Architecture (from poster + Fig.4):
  - Encoder: Conv2D(in→64) → ResBlocks(64) → ResBlock+Down(128) →
             ResBlocks(128) → ResBlock+Down(256) → ResBlocks(256) →
             ResBlock+Down(512) → ResBlocks(512) → ResBlock+Down(512)
  - Bottleneck: ResBlock(512) + SelfAttention + ResBlock(512)
  - Decoder: mirrors encoder with concatenative skip connections.
  - Out: GroupNorm → SiLU → zero_module(Conv2D(64 → 2))
        First output channel = r̂ (residual / nDSM)
        Second output channel = ℓ (per-pixel confidence logits)
  - FiLM timestep conditioning per ResBlock (γ embedding).

Defaults give ~62.6M parameters at image_size=256 (paper §8.1).

Adapted from openai/guided-diffusion via Palette.
"""
from __future__ import annotations
from abc import abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import normalization, zero_module, gamma_embedding


class _EmbedBlock(nn.Module):
    """Module whose forward takes (x, emb)."""
    @abstractmethod
    def forward(self, x, emb): ...


class _EmbedSequential(nn.Sequential, _EmbedBlock):
    """Sequential that propagates `emb` only into _EmbedBlock children."""
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, _EmbedBlock) else layer(x)
        return x


class _Upsample(nn.Module):
    def __init__(self, channels, use_conv=True, out_ch=None):
        super().__init__()
        self.channels = channels
        out_ch = out_ch or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class _Downsample(nn.Module):
    def __init__(self, channels, use_conv=True, out_ch=None):
        super().__init__()
        out_ch = out_ch or channels
        if use_conv:
            self.op = nn.Conv2d(channels, out_ch, 3, stride=2, padding=1)
        else:
            assert channels == out_ch
            self.op = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.op(x)


class _ResBlock(_EmbedBlock):
    """Pre-activation ResBlock with FiLM γ-conditioning, optionally
    folding up/downsampling into the residual path."""

    def __init__(self, channels, emb_ch, dropout=0.0, out_ch=None,
                 use_scale_shift_norm=True, up=False, down=False):
        super().__init__()
        out_ch = out_ch or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up or down

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_ch, 3, padding=1),
        )

        if up:
            self.h_upd = _Upsample(channels, use_conv=False)
            self.x_upd = _Upsample(channels, use_conv=False)
        elif down:
            self.h_upd = _Downsample(channels, use_conv=False)
            self.x_upd = _Downsample(channels, use_conv=False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # FiLM: emb → 2*out_ch (scale, shift) when use_scale_shift_norm
        emb_out_dim = 2 * out_ch if use_scale_shift_norm else out_ch
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_ch, emb_out_dim),
        )

        self.out_layers = nn.Sequential(
            normalization(out_ch),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
        )

        if out_ch == channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(channels, out_ch, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while emb_out.dim() < h.dim():
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip(x) + h


class _Attention(nn.Module):
    """Spatial self-attention block (multi-head). Applied at the
    bottleneck only by default — paper Fig.4."""

    def __init__(self, channels, num_heads=4, num_head_channels=-1):
        super().__init__()
        if num_head_channels == -1:
            assert channels % num_heads == 0
            head_dim = channels // num_heads
        else:
            assert channels % num_head_channels == 0
            num_heads = channels // num_head_channels
            head_dim = num_head_channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        flat = x.reshape(b, c, h * w)
        qkv = self.qkv(self.norm(flat))                       # [B, 3C, HW]
        qkv = qkv.view(b * self.num_heads, 3 * self.head_dim, h * w)
        q, k, v = qkv.split(self.head_dim, dim=1)
        scale = 1.0 / math.sqrt(math.sqrt(self.head_dim))
        # More numerically stable than scaling once after the matmul
        attn = torch.einsum('bct,bcs->bts', q * scale, k * scale)
        attn = torch.softmax(attn.float(), dim=-1).type(attn.dtype)
        out = torch.einsum('bts,bcs->bct', attn, v)
        out = out.reshape(b, c, h * w)
        out = self.proj_out(out)
        return (flat + out).reshape(b, c, h, w)


class GrounDiffUNet(nn.Module):
    """GrounDiff denoiser (paper §3.2 Fig.4).

    in_channel: 2 — (DSM, noisy_DTM) channel-concatenated.
    out_channel: 2 — (residual r̂, confidence logit ℓ).
    inner_channel: 64 — base width.
    channel_mults: (1, 2, 4, 8) — width per stage.
    res_blocks: 2 per stage.
    attn_res: {16} — attention only at the bottleneck level (paper).

    With image_size=256, channel_mults=(1,2,4,8), this gives spatial
    resolutions 256→128→64→32→16, and ~62.6M parameters (paper §8.1).
    """

    def __init__(self, *, image_size: int = 256,
                 in_channel: int = 2,
                 out_channel: int = 2,
                 inner_channel: int = 64,
                 channel_mults=(1, 2, 4, 8),
                 attn_res=(16,),
                 res_blocks: int = 2,
                 num_head_channels: int = 32,
                 dropout: float = 0.0,
                 use_scale_shift_norm: bool = True):
        super().__init__()
        self.image_size = image_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.inner_channel = inner_channel

        # γ → embedding → MLP
        emb_ch = inner_channel * 4
        self.cond_embed = nn.Sequential(
            nn.Linear(inner_channel, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch),
        )

        ch = int(channel_mults[0] * inner_channel)
        input_ch = ch
        self.input_blocks = nn.ModuleList(
            [_EmbedSequential(nn.Conv2d(in_channel, ch, 3, padding=1))]
        )
        skip_chs = [ch]
        ds = 1                                    # downsample factor
        for level, mult in enumerate(channel_mults):
            for _ in range(res_blocks):
                layers = [
                    _ResBlock(ch, emb_ch, dropout,
                              out_ch=int(mult * inner_channel),
                              use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = int(mult * inner_channel)
                if ds in attn_res:
                    layers.append(
                        _Attention(ch, num_head_channels=num_head_channels))
                self.input_blocks.append(_EmbedSequential(*layers))
                skip_chs.append(ch)
            if level != len(channel_mults) - 1:
                self.input_blocks.append(
                    _EmbedSequential(
                        _ResBlock(ch, emb_ch, dropout, out_ch=ch,
                                  use_scale_shift_norm=use_scale_shift_norm,
                                  down=True)
                    )
                )
                skip_chs.append(ch)
                ds *= 2

        # Bottleneck: Res → Attn → Res
        self.middle_block = _EmbedSequential(
            _ResBlock(ch, emb_ch, dropout,
                      use_scale_shift_norm=use_scale_shift_norm),
            _Attention(ch, num_head_channels=num_head_channels),
            _ResBlock(ch, emb_ch, dropout,
                      use_scale_shift_norm=use_scale_shift_norm),
        )

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(res_blocks + 1):
                ich = skip_chs.pop()
                layers = [
                    _ResBlock(ch + ich, emb_ch, dropout,
                              out_ch=int(inner_channel * mult),
                              use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = int(inner_channel * mult)
                if ds in attn_res:
                    layers.append(
                        _Attention(ch, num_head_channels=num_head_channels))
                if level and i == res_blocks:
                    layers.append(
                        _ResBlock(ch, emb_ch, dropout, out_ch=ch,
                                  use_scale_shift_norm=use_scale_shift_norm,
                                  up=True))
                    ds //= 2
                self.output_blocks.append(_EmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, out_channel, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor, gammas: torch.Tensor):
        """Forward pass.

        Args:
            x:      [B, in_channel, H, W] — concat of cond and noisy target
            gammas: [B] — γ_t = ᾱ_t for the batch (Palette convention)

        Returns:
            [B, out_channel, H, W] — first channel r̂, second ℓ.
        """
        emb = self.cond_embed(gamma_embedding(gammas.view(-1),
                                              self.inner_channel))
        h = x
        skips = []
        for blk in self.input_blocks:
            h = blk(h, emb)
            skips.append(h)
        h = self.middle_block(h, emb)
        for blk in self.output_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = blk(h, emb)
        return self.out(h)

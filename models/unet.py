"""GrounDiff U-Net backbone.

Architectural pedigree:
  OpenAI guided-diffusion → Palette (Saharia et al. 2022) → official
  GrounDiff (Dhaouadi et al., WACV 2026) → this. The structural code —
  pre-activation ResBlocks, FiLM scale-shift, GroupNorm32 with fp32
  upcast, SiLU, bottleneck self-attention, zero-init final conv,
  γ-encoded conditioning — is Palette / guided-diffusion lineage.
  We train from scratch in pixel space on the 4-channel conditioning
  input.

Architecture summary (paper §8.1, image_size=256, 62.6M params):
  - Encoder: Conv2D(4 → C) → ResBlocks(C) → ResBlock+Down(2C) →
             ResBlocks(2C) → ResBlock+Down(4C) → ResBlocks(4C) →
             ResBlock+Down(8C) → ResBlocks(8C)
             (channel_mults=(1,2,4,8), 4 levels, 3 downsamples)
  - Bottleneck: ResBlock + SelfAttention + ResBlock at 32×32 (for 256²
                input) = 1024 attention tokens per head.
  - Decoder: mirrors encoder with concatenative skip connections.
  - Out: GroupNorm → SiLU → zero_module(Conv2D(C → 2))
        First output channel = r̂ (residual / nDSM)
        Second output channel = ℓ (per-pixel confidence logits)
  - FiLM γ-conditioning per ResBlock (γ embedding → scale, shift).

Why these scales (256² tile, 4 channels):
  - Paper §7.1: 256×256 tiles, single-scale operation.
  - 3 downsamples: 256 → 128 → 64 → 32. The 32×32 bottleneck gives
    1024 attention tokens — receptive field spans the whole tile.
  - inner_channel=64, mults=(1,2,4,8) puts the deepest level at 512
    channels. Total ~62.6M params (matches paper §8.1 exactly with
    in_channels=2; our in_channels=4 adds ~1500 stem-conv weights,
    negligible).

Forward signature:
    forward(x: [B, 4, H, W], gammas: [B]) -> [B, 2, H, W]
"""
from __future__ import annotations

from abc import abstractmethod
import math

import torch
import torch.nn as nn

from .nn_unet import normalization, zero_module, gamma_embedding


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
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
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
    folding up/down-sampling into the residual path."""

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
    bottleneck only by default.

    Uses `F.scaled_dot_product_attention` which on Blackwell dispatches
    to FlashAttention (or memory-efficient attention) automatically. At
    our bottleneck of 64×64 = 4096 tokens with 4-16 heads, this is much
    faster and lower-memory than a manual einsum because it fuses the
    QK^T → softmax → AV chain into one kernel and avoids materialising
    the [4096, 4096] attention matrix in HBM.
    """

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
        n = h * w
        flat = x.reshape(b, c, n)
        qkv = self.qkv(self.norm(flat))                       # [B, 3C, N]
        # Build Q, K, V each as [B, num_heads, N, head_dim] with
        # last-dim-contiguous strides. The contiguous() call materialises
        # the transpose so Inductor doesn't have to insert a layout-
        # conversion copy at every compiled SDPA call. Without it we get
        # the `Layout conflict detected for bufNNN: ... but layout is
        # frozen to [stride 1 on middle dim]` warnings during compile,
        # and the resulting code runs an extra copy per attention.
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, n)
        q, k, v = qkv.unbind(dim=1)                           # each [B, H, D, N]
        q = q.transpose(-1, -2).contiguous()                   # [B, H, N, D]
        k = k.transpose(-1, -2).contiguous()
        v = v.transpose(-1, -2).contiguous()
        out = nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, n)           # [B, C, N]
        out = self.proj_out(out)
        return (flat + out).reshape(b, c, h, w)


class GrounDiffUNet(nn.Module):
    """GrounDiff denoiser (UNet backbone).

    in_channels: 4 — [g_t, dsm_max, dsm_min, dsm_mean].
    out_channels: 2 — [r̂, ℓ] (residual + confidence logit).

    Defaults (paper §8.1, image_size=256):
        inner_channel=64, channel_mults=(1, 2, 4, 8), 4 levels.
        Bottleneck at 1/8 spatial (32×32 for 256² input) with
        self-attention. ~62.6M parameters at in_channels=2;
        in_channels=4 adds ~1500 stem-conv weights (negligible).

    `attn_at_levels=()` is bottleneck-only attention (paper). The
    bottleneck's always-on residual-attention-residual stack is built
    regardless of this list — adding levels here ADDS attention at
    shallower encoder/decoder depths, with quadratic memory cost in
    the attention spatial extent.
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        out_channels: int = 2,
        inner_channel: int = 64,
        channel_mults=(1, 2, 4, 8),
        attn_at_levels=(),
        res_blocks: int = 2,
        num_head_channels: int = 32,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
        zero_init_output: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_channel = inner_channel
        # Store for downstream consumers (e.g. PrioStitch needs to know
        # the spatial divisibility constraint for resampling). Each level
        # except the last in channel_mults adds a stride-2 downsample, so
        # input H/W must be divisible by 2 ** (len(channel_mults) - 1).
        self.channel_mults = tuple(channel_mults)
        self.grid_multiple = 2 ** (len(self.channel_mults) - 1)

        # γ-embedding MLP: matches Palette / guided-diffusion convention.
        emb_ch = inner_channel * 4
        self.cond_embed = nn.Sequential(
            nn.Linear(inner_channel, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch),
        )

        # Translate level indices (0-based depth from input) into the
        # `ds` counter (downsample factor). attn_at_levels=() leaves
        # attention bottleneck-only (paper §3.2). Passing e.g. (3,)
        # would ADD attention at encoder/decoder level 3 (ds=8) on top
        # of the bottleneck.
        attn_ds = {2 ** lvl for lvl in attn_at_levels}

        ch = int(channel_mults[0] * inner_channel)
        input_ch = ch
        self.input_blocks = nn.ModuleList(
            [_EmbedSequential(nn.Conv2d(in_channels, ch, 3, padding=1))]
        )
        skip_chs = [ch]
        ds = 1
        for level, mult in enumerate(channel_mults):
            for _ in range(res_blocks):
                layers = [
                    _ResBlock(ch, emb_ch, dropout,
                              out_ch=int(mult * inner_channel),
                              use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = int(mult * inner_channel)
                if ds in attn_ds:
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

        # Bottleneck always has attention regardless of attn_at_levels.
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
                if ds in attn_ds:
                    layers.append(
                        _Attention(ch, num_head_channels=num_head_channels))
                if level and i == res_blocks:
                    layers.append(
                        _ResBlock(ch, emb_ch, dropout, out_ch=ch,
                                  use_scale_shift_norm=use_scale_shift_norm,
                                  up=True))
                    ds //= 2
                self.output_blocks.append(_EmbedSequential(*layers))

        out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        if zero_init_output:
            out_conv = zero_module(out_conv)
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            out_conv,
        )

    def forward(self, x: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x:      [B, in_channels, H, W] — [g_t, dsm_max, dsm_min,
                    dsm_mean] concatenated (4 channels).
            gammas: [B] — γ (cumulative noise level).

        Returns:
            [B, out_channels, H, W] — channel 0 r̂, channel 1 ℓ.
        """
        # Channels-last (NHWC) layout. cuDNN's tensor-core conv kernels
        # are ~10-30% faster when both weights and activations live in
        # NHWC. Weight conversion happens once in train.py
        # (`model.to(memory_format=torch.channels_last)`); we convert
        # input activations here so every forward — training, eval,
        # PrioStitch inference — gets the fast path. On CPU this is a
        # no-op stride change; on CUDA it's a single async copy whose
        # cost is dwarfed by the conv speedup.
        if x.is_cuda:
            x = x.to(memory_format=torch.channels_last)
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

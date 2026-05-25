"""GrounDiff (raster) top-level wrapper.

Bundles the UNet denoiser + diffusion + loss into one `nn.Module` with
`training_step` and `infer` methods that hide the multi-channel input
plumbing.

Backbone: Palette-derived / OpenAI-guided-diffusion UNet (the same
lineage as stage 1 GrounDiff), sized up for 1024² tiles + 5-channel
input. ~80M params. Marigold's empirical validation that this kind of
UNet is still SOTA for dense pixel-aligned regression in 2024-2025 is
the reason we chose this over a DiT — but the code itself is Palette
heritage, not a Marigold port (Marigold operates in SD's latent space,
which is RGB-trained and inappropriate for LiDAR rasters).

Conditioning channels:
    [g_t, dsm_max, dsm_min]   (3 channels — matches official defra.json `use_min_dsm: true`)
Output channels:
    [r̂, ℓ]                                         (2 channels)

The gating function combines r̂, ℓ and `s = dsm_max` to produce the
final gated DTM ĝ used by the loss and as the predicted clean image
during sampling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .diffusion import GrounDiffDiffusion
from .gating import gating
from .losses import groundiff_loss
from .unet import GrounDiffUNet


class GrounDiffRaster(nn.Module):
    """GrounDiff (raster) — UNet-backed pixel-space diffusion.

    The denoiser and the diffusion are decoupled — only the denoiser
    has trainable parameters; the diffusion just holds the noise
    schedule. Checkpoints save only denoiser weights.

    Backwards compatibility: pre-existing checkpoints used `dit` as the
    state-dict key. The loader handles both `net` and `dit` keys and
    silently falls back to a raw state-dict if neither is present.
    """

    def __init__(
        self,
        *,
        backbone_kwargs: Optional[dict] = None,
        diffusion_kwargs: Optional[dict] = None,
        loss_kwargs: Optional[dict] = None,
        # Legacy aliases. Old configs / scripts may pass `dit_kwargs` or
        # `backbone="unet"` — we accept them silently for compatibility.
        dit_kwargs: Optional[dict] = None,
        backbone: Optional[str] = None,
    ):
        super().__init__()
        if backbone_kwargs is None and dit_kwargs is not None:
            backbone_kwargs = dit_kwargs
        backbone_kwargs = backbone_kwargs or {}
        diffusion_kwargs = diffusion_kwargs or {}
        loss_kwargs = dict(loss_kwargs or {})

        # `alpha_metres` is a dataset-side config (used in preprocess to
        # build M_α). Drop from loss kwargs and any `_note_*` doc keys.
        loss_kwargs.pop("alpha_metres", None)
        for k in list(loss_kwargs.keys()):
            if k.startswith("_"):
                loss_kwargs.pop(k)
        self.loss_kwargs = loss_kwargs

        if backbone is not None and backbone != "unet":
            raise ValueError(
                f"This build only supports backbone='unet'; got {backbone!r}. "
                f"The CascadeDiT backbone was removed when committing to "
                f"UNet — grab the previous stage2_raster.zip if you need it.")
        self.backbone_name = "unet"
        # `self.net` is the canonical name; `self.dit` is kept as an
        # alias so old scripts and old checkpoint loaders keep working.
        self.net = GrounDiffUNet(**backbone_kwargs)
        self.dit = self.net
        self.diffusion = GrounDiffDiffusion(**diffusion_kwargs)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def training_step(
        self,
        dsm_max: torch.Tensor,
        dsm_min: torch.Tensor,
        dsm_mean: torch.Tensor,
        dsm_mask: torch.Tensor,
        dtm: torch.Tensor,
        valid: torch.Tensor,
        m_alpha: torch.Tensor,
    ):
        """One paper-faithful training pass.

        All inputs in NORMALISED frame:
            dsm_max:  [B, 1, H, W]   per-tile normalised to [-1, +1]
            dsm_min:  [B, 1, H, W]   per-tile normalised (uses tile_stats
                                     of dsm_max so the three DSM channels
                                     share a frame)
            dsm_mean: [B, 1, H, W]   per-tile normalised
            dsm_mask: [B, 1, H, W]   binary {0, 1} (1 = had at least one
                                     return in that cell)
            dtm:      [B, 1, H, W]   GT DTM, per-tile normalised
            valid:    [B, 1, H, W]   {0, 1} pixels-with-GT mask
            m_alpha:  [B, 1, H, W]   M_α target built in metres-space at
                                     preprocess time (α = 0.20 m).

        Returns:
            (total_loss, metrics_dict)
        """
        device = dsm_max.device
        bs = dsm_max.shape[0]

        # Sample γ ~ Palette-style continuous distribution, then q_sample
        # the GT to produce the noisy input g_t.
        gammas = self.diffusion.sample_gammas(bs, device)
        noise = torch.randn_like(dtm)
        g_t = self.diffusion.q_sample(dtm, gammas, noise=noise)

        # Channel order: [g_t, dsm_max, dsm_min]  (3 channels)
        # Matches official defra.json `use_min_dsm: true`.
        # dsm_mask is NOT fed as an input. It's still loaded by the
        # dataset for downstream uses (validation metric splits etc.)
        # but is not used in the loss either. Loss masking uses
        # valid_gt only (see below) — paper-faithful "exclude no-data
        # regions" without any arbitrary additional filtering.
        x = torch.cat([g_t, dsm_max, dsm_min], dim=1)
        out = self.net(x, gammas)                  # [B, 2, H, W]
        r_hat, logit = out[:, 0:1], out[:, 1:2]
        g_hat = gating(r_hat, logit, dsm_max)

        # Pass `valid` mask to the loss. Definition: a pixel is valid
        # if (DSM exists) AND (DTM exists) at that location. Paper §7.2
        # Eq.15 intersection: a single mask `m` defines valid pixels in
        # both s and g simultaneously. Pixels with neither / only-one
        # source are excluded — they'd contribute spurious gradients
        # (DSM-only has no GT target; DTM-only has sentinel DSM input).
        # Paper's dense training tiles have valid all-ones; this reduces
        # to no masking on their setup.
        loss, metrics = groundiff_loss(
            g_hat=g_hat, g_target=dtm, logit=logit,
            valid=valid, m_alpha=m_alpha,
            **self.loss_kwargs)
        return loss, metrics

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def infer(
        self,
        dsm_max: torch.Tensor,
        dsm_min: torch.Tensor,
        dsm_mean: torch.Tensor | None = None,
        dsm_mask: torch.Tensor | None = None,
        *,
        init: str = "noisy_dsm",
        prior_dtm: Optional[torch.Tensor] = None,
        return_logit: bool = False,
    ):
        """Reverse diffusion to produce the predicted DTM.

        Args:
            dsm_max, dsm_min, dsm_mask: see `training_step`.
            dsm_mean: ACCEPTED FOR BACKWARD COMPATIBILITY but ignored —
                      the network input is 3-channel [g_t, dsm_max, dsm_min]
                      (official defra.json `use_min_dsm: true`). dsm_mean
                      is no longer concatenated. Callers may still pass it.
            init: 'noisy_dsm' (paper default) | 'gaussian' | 'priostitch'.
            prior_dtm: required when init='priostitch' — the upsampled
                       global prior, in the same normalised frame as dsm_max.
            return_logit: also return the final-step logit ℓ.

        Returns:
            [B, 1, H, W] predicted DTM in normalised [-1, +1] frame.
            If return_logit: (dtm, logit).
        """
        _ = dsm_mean  # accepted but unused (3-channel network)
        if init == "priostitch":
            assert prior_dtm is not None, \
                "init='priostitch' requires prior_dtm"
            return self.diffusion.sample_from_prior(
                self.net, dsm_max=dsm_max, dsm_min=dsm_min,
                dsm_mask=dsm_mask,
                prior_dtm=prior_dtm, return_logit=return_logit)
        return self.diffusion.sample(
            self.net, dsm_max=dsm_max, dsm_min=dsm_min,
            dsm_mask=dsm_mask,
            init=init, return_logit=return_logit)

    # ------------------------------------------------------------------ #
    #  Checkpoint I/O
    # ------------------------------------------------------------------ #

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write under the new canonical key `net` plus the legacy `dit`
        # key (a shallow copy of the same state-dict) for backward
        # compatibility with older loaders.
        sd = self.net.state_dict()
        torch.save(dict(net=sd, dit=sd, backbone=self.backbone_name), path)

    def load(self, path, *, strict: bool = True, map_location=None):
        d = torch.load(path, map_location=map_location, weights_only=True)
        # Try the new key first, then the legacy `dit` key, then assume
        # it's a raw state-dict.
        if "net" in d:
            self.net.load_state_dict(d["net"], strict=strict)
        elif "dit" in d:
            self.net.load_state_dict(d["dit"], strict=strict)
        else:
            self.net.load_state_dict(d, strict=strict)
        return self

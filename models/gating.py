"""GrounDiff gating function (paper §3.2, Eq. 5).

    G(r̂, ℓ, s) = σ(ℓ) ⊙ s + (1 − σ(ℓ)) ⊙ (s − r̂)

Where:
  s   is the conditioning DSM (max-z channel, normalised)
  r̂   is the network's residual prediction (predicted nDSM)
  ℓ   is the network's per-pixel confidence logit
  σ() is the sigmoid

Reading: where the model is confident a pixel is ground (σ(ℓ)→1) the
output equals the DSM unchanged; where the model is confident the pixel
is non-ground (σ(ℓ)→0) the output is DSM minus the predicted nDSM. The
sigmoid linearly blends in between.

This is the ONLY place gating logic lives. Both training (training_step)
and inference (sample / sample_from_prior) call this.
"""
from __future__ import annotations

import torch


def gating(r_hat: torch.Tensor,
           logit: torch.Tensor,
           dsm: torch.Tensor) -> torch.Tensor:
    """Apply Eq. 5 to predict the clean DTM.

    Args:
        r_hat: [B, 1, H, W]  predicted residual (nDSM)
        logit: [B, 1, H, W]  raw confidence logits
        dsm:   [B, 1, H, W]  conditioning DSM (max-z, normalised)

    Returns:
        g_hat: [B, 1, H, W]  gated DTM in the same frame as `dsm`.
    """
    sig = torch.sigmoid(logit)
    return sig * dsm + (1.0 - sig) * (dsm - r_hat)

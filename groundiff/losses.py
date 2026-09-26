"""GrounDiff training loss (paper Eq. 11-14) and the ResDepth L1 loss.

All GrounDiff terms are averaged over valid pixels only (supplement §7.2:
invalid regions are excluded from the loss). Inputs are [B, 1, H, W].
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    lam_l1: float = 1.0      # λ1
    lam_l2: float = 1.0      # λ2
    lam_grad: float = 0.1    # λ∇
    lam_conf: float = 0.1    # λc


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def gradient_magnitude(x: torch.Tensor):
    """|∇x| from forward differences, shape [B, 1, H-1, W-1]."""
    dx = x[..., :-1, 1:] - x[..., :-1, :-1]
    dy = x[..., 1:, :-1] - x[..., :-1, :-1]
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def groundiff_loss(g0_hat: torch.Tensor, logit: torch.Tensor, g0: torch.Tensor,
                   m_alpha: torch.Tensor, valid: torch.Tensor, cfg: LossConfig) -> dict:
    valid = valid.to(g0_hat.dtype)
    err = g0_hat - g0
    l1 = _masked_mean(err.abs(), valid)                                    # Eq. 12
    l2 = _masked_mean(err * err, valid)                                    # Eq. 12
    # Eq. 13 on gradient magnitudes; a difference is valid only when all
    # three pixels it touches are valid.
    gvalid = valid[..., :-1, :-1] * valid[..., :-1, 1:] * valid[..., 1:, :-1]
    lgrad = _masked_mean((gradient_magnitude(g0_hat) - gradient_magnitude(g0)).abs(), gvalid)
    bce = F.binary_cross_entropy_with_logits(logit, m_alpha.to(logit.dtype), reduction="none")
    lconf = _masked_mean(bce, valid)                                       # Eq. 14
    total = cfg.lam_l1 * l1 + cfg.lam_l2 * l2 + cfg.lam_grad * lgrad + cfg.lam_conf * lconf  # Eq. 11
    return {"loss": total, "l1": l1.detach(), "l2": l2.detach(),
            "grad": lgrad.detach(), "conf": lconf.detach()}


def resdepth_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                  std: torch.Tensor) -> dict:
    """ResDepth: L1 in metres over valid pixels (lib/Trainer.py
    `_compute_denormalized_loss`: both rasters are de-normalised by the std
    before the loss; the mean cancels)."""
    valid = valid.to(pred.dtype)
    l1 = _masked_mean((pred - target).abs() * std.view(-1, 1, 1, 1), valid)
    return {"loss": l1, "l1": l1.detach()}

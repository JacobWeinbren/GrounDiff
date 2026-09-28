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
    # "normalised": per-tile [-1, 1] units as in the paper; "metres": the same
    # terms in metres, so high-relief tiles are not under-weighted
    units: str = "normalised"
    # "laplace": the L1 term becomes the Laplace negative log-likelihood |e| / b + log b with a predicted
    # per-pixel scale b (model.aleatoric): the median is still the optimum (as for L1, which Noise2Noise
    # needs), and b learns how noisy the labels are there (heteroscedastic aleatoric uncertainty)
    nll: str = "none"


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def gradient_magnitude(x: torch.Tensor):
    """|∇x| from forward differences, shape [B, 1, H-1, W-1]."""
    dx = x[..., :-1, 1:] - x[..., :-1, :-1]
    dy = x[..., 1:, :-1] - x[..., :-1, :-1]
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def groundiff_loss(g0_hat: torch.Tensor, logit: torch.Tensor, g0: torch.Tensor,
                   m_alpha: torch.Tensor, valid: torch.Tensor, cfg: LossConfig,
                   half_scale: torch.Tensor | None = None, log_b: torch.Tensor | None = None) -> dict:
    """half_scale [B]: metres per normalised unit (scale / 2), needed for units="metres".
    log_b: predicted log Laplace scale (normalised units), used when cfg.nll == "laplace"."""
    valid = valid.to(g0_hat.dtype)
    if cfg.units == "metres":
        if half_scale is None:
            raise ValueError("units='metres' needs half_scale")
        hs = half_scale.view(-1, 1, 1, 1).to(g0_hat.dtype)
        g0_hat, g0 = g0_hat * hs, g0 * hs
        if log_b is not None:
            log_b = log_b + torch.log(hs)
    err = g0_hat - g0
    l1 = _masked_mean(err.abs(), valid)                                    # Eq. 12
    if cfg.nll == "laplace":
        if log_b is None:
            raise ValueError("loss.nll='laplace' needs model.aleatoric=true")
        lb = log_b.clamp(-9.0, 3.0)
        nll = _masked_mean(err.abs() * torch.exp(-lb) + lb, valid)
    elif cfg.nll != "none":
        raise ValueError(f"loss.nll must be none or laplace, got {cfg.nll!r}")
    l2 = _masked_mean(err * err, valid)                                    # Eq. 12
    # Eq. 13 on gradient magnitudes; a difference is valid only when all
    # three pixels it touches are valid.
    gvalid = valid[..., :-1, :-1] * valid[..., :-1, 1:] * valid[..., 1:, :-1]
    lgrad = _masked_mean((gradient_magnitude(g0_hat) - gradient_magnitude(g0)).abs(), gvalid)
    bce = F.binary_cross_entropy_with_logits(logit, m_alpha.to(logit.dtype), reduction="none")
    lconf = _masked_mean(bce, valid)                                       # Eq. 14
    first = nll if cfg.nll == "laplace" else l1
    total = cfg.lam_l1 * first + cfg.lam_l2 * l2 + cfg.lam_grad * lgrad + cfg.lam_conf * lconf  # Eq. 11
    out = {"loss": total, "l1": l1.detach(), "l2": l2.detach(), "grad": lgrad.detach(), "conf": lconf.detach()}
    if cfg.nll == "laplace":
        out["nll"] = nll.detach()
    return out


def resdepth_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                  std: torch.Tensor) -> dict:
    """ResDepth: L1 in metres over valid pixels (lib/Trainer.py
    `_compute_denormalized_loss`: both rasters are de-normalised by the std
    before the loss; the mean cancels)."""
    valid = valid.to(pred.dtype)
    l1 = _masked_mean((pred - target).abs() * std.view(-1, 1, 1, 1), valid)
    return {"loss": l1, "l1": l1.detach()}

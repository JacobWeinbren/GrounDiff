"""GrounDiff loss (paper §3.2 Eq. 11-14).

    L = λ₁ · L₁ + λ₂ · L₂ + λ_∇ · L_∇ + λ_c · L_c

    L₁ = ||ĝ − g||₁                           (paper Eq. 12)
    L₂ = ||ĝ − g||₂²                           (paper Eq. 12)
    L_∇ = || |∇ĝ|₂ − |∇g|₂ ||₁                 (paper Eq. 13, gradient *magnitude*)
    L_c = BCE(σ(ℓ), M_α)                       (paper Eq. 14)

Defaults (paper §7.3): λ₁ = λ₂ = 1.0, λ_∇ = λ_c = 0.1.

M_α is computed in METRES at the dataset level (paper does not specify
a numerical value of α; we use Sithole & Vosselman 2003 §4.2.1 α=0.20 m
for cross-tile consistency). This module never recomputes M_α — it
always uses the pre-built binary mask passed in `m_alpha`.

Crucially, L_∇ uses gradient *magnitudes*, not full gradient vectors
(paper §3.2: "advantageous since ground-truth DTMs often use
triangulation beneath non-ground structures, creating arbitrary
orientation patterns").

The loss operates on the GATED prediction ĝ (output of gating()), not on
r̂ directly. The logit ℓ enters via the gated ĝ (in L₁/L₂/L_∇) and via
L_c (BCE term).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _grad_magnitude(x: torch.Tensor) -> torch.Tensor:
    """L2 magnitude of the spatial gradient, returned at the same H,W as x.

    We pad the per-axis differences back to (H, W) with zeros — i.e. the
    gradient of a constant boundary is zero — to keep tensor shapes
    aligned without inventing edge values.
    """
    dx = x[..., :, 1:] - x[..., :, :-1]     # [B, 1, H, W-1]
    dy = x[..., 1:, :] - x[..., :-1, :]     # [B, 1, H-1, W]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def groundiff_loss(
    g_hat: torch.Tensor,
    g_target: torch.Tensor,
    logit: torch.Tensor,
    dsm: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    m_alpha: torch.Tensor | None = None,
    *,
    alpha: float = 0.05,
    lam_l1: float = 1.0,
    lam_l2: float = 1.0,
    lam_grad: float = 0.1,
    lam_conf: float = 0.1,
):
    """Paper-faithful GrounDiff loss. Signature matches the official
    `clean rebuild` reference (groundiff/models/losses.py) for API
    parity: same call → same numerics.

    Args:
        g_hat:    [B, 1, H, W]  GATED predicted DTM (output of gating()).
        g_target: [B, 1, H, W]  GT DTM in the same (normalised) frame.
        logit:    [B, 1, H, W]  raw confidence logits from the UNet.
        dsm:      [B, 1, H, W]  conditioning DSM (normalised). Only used
                   by the `alpha` fallback when `m_alpha` is None.
        valid:    [B, 1, H, W] or [B, H, W] or None — pixels to include.
                   None ⇒ all pixels counted.
        m_alpha:  [B, 1, H, W] or [B, H, W] — binary M_α mask in {0,1}.
                   When supplied (preferred path), used directly. When
                   None, falls back to `|dsm − g_target| < alpha`
                   in NORMALISED units — unit-inconsistent across tiles,
                   retained only for API parity / unit tests.
        alpha:    Legacy fallback threshold IN NORMALISED UNITS, used
                   only when `m_alpha is None and dsm is not None`.
                   Paper §7.3 does not specify a numerical α; prefer
                   precomputing `m_alpha` in metres at preprocess.
        lam_*:    Loss weights — paper §7.3 defaults 1, 1, 0.1, 0.1.

    Returns:
        (total_loss, dict of detached per-term losses)
    """
    if valid is None:
        v = torch.ones_like(g_hat)
    else:
        v = valid.to(g_hat.dtype)
        if v.dim() == 3:
            v = v.unsqueeze(1)
    n_valid = v.sum().clamp(min=1.0)

    diff = g_hat - g_target
    l1 = (diff.abs() * v).sum() / n_valid
    l2 = (diff.pow(2) * v).sum() / n_valid

    # L_∇ on gradient magnitudes
    grad_pred = _grad_magnitude(g_hat)
    grad_gt = _grad_magnitude(g_target)
    l_grad = ((grad_pred - grad_gt).abs() * v).sum() / n_valid

    # L_c: BCE(σ(ℓ), M_α). Prefer precomputed mask in metres-space.
    if m_alpha is not None:
        ma = m_alpha.to(g_hat.dtype)
        if ma.dim() == 3:
            ma = ma.unsqueeze(1)
    elif dsm is not None:
        # Reference-API fallback: recompute in normalised units. Not
        # paper-faithful for cross-tile consistency.
        with torch.no_grad():
            ma = ((dsm - g_target).abs() < alpha).to(g_hat.dtype)
    else:
        raise ValueError(
            "groundiff_loss needs `m_alpha` (preferred) or `dsm` "
            "(legacy fallback). Got neither.")

    bce = F.binary_cross_entropy_with_logits(logit, ma, reduction="none")
    l_conf = (bce * v).sum() / n_valid

    total = (lam_l1 * l1 + lam_l2 * l2
             + lam_grad * l_grad + lam_conf * l_conf)

    return total, dict(
        l1=l1.detach(),
        l2=l2.detach(),
        grad=l_grad.detach(),
        conf=l_conf.detach(),
        loss=total.detach(),
    )

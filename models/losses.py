"""GrounDiff loss (paper §3.2 Eq.11-14).

L = λ₁·L₁ + λ₂·L₂ + λ_∇·L_∇ + λ_c·L_c

L₁ = ||ĝ - g||₁                  (paper Eq.12, regression)
L₂ = ||ĝ - g||₂²                 (paper Eq.12, smooths roads / under-building areas)
L_∇ = ||  |∇ĝ|₂ − |∇g|₂  ||₁     (paper Eq.13, edge-aware on gradient *magnitude*)
L_c = BCE(σ(ℓ), M_α)             (paper Eq.14, M_α = 1[|s - g| < α])

Defaults: λ₁ = λ₂ = 1.0, λ_∇ = λ_c = 0.1 (paper §7.3).

M_α is computed in METRES at the dataset level (paper does not specify
a numerical α; we use the Sithole-Vosselman 2003 §4.2.1 convention
α_metres = 0.20m for cross-tile consistency). The dataset hands a
precomputed binary mask to this loss to avoid scale-dependence: with
per-tile normalisation, "α=0.05 in normalised units" means different
metres per tile, which is unprincipled.

Crucial: L_∇ uses gradient *magnitudes*, not full gradient vectors.
Paper §3.2: "advantageous since ground-truth DTMs often use triangulation
beneath non-ground structures, creating arbitrary orientation patterns".

The loss operates on the GATED prediction ĝ = G(r̂, ℓ, s), not on r̂
directly — that's where the residual r̂ and the logit ℓ jointly drive
gradients. See models/groundiff.py for the wiring.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _grad_magnitude(x: torch.Tensor) -> torch.Tensor:
    """L2 magnitude of the spatial gradient. Returns same H,W via
    'replicate' boundary so we don't lose mass on the edge column/row.

    Sobel on each axis would also work; we use simple finite differences
    consistent with `torch.gradient` semantics (paper doesn't specify
    the operator, only that magnitudes are taken).
    """
    # x: [B, 1, H, W]
    # dx along width (last dim), dy along height (second-last dim)
    dx = x[..., :, 1:] - x[..., :, :-1]                # [B, 1, H, W-1]
    dy = x[..., 1:, :] - x[..., :-1, :]                # [B, 1, H-1, W]
    # Pad each back to [B, 1, H, W] using replicate so edge gradients
    # are 0 (gradient of a constant) — mathematically the right boundary.
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def groundiff_loss(g_hat: torch.Tensor, g_target: torch.Tensor,
                   logit: torch.Tensor, dsm: torch.Tensor,
                   valid: torch.Tensor | None = None,
                   m_alpha: torch.Tensor | None = None,
                   *, alpha: float = 0.05,
                   lam_l1: float = 1.0, lam_l2: float = 1.0,
                   lam_grad: float = 0.1, lam_conf: float = 0.1):
    """Compute the four-component GrounDiff loss.

    Args:
        g_hat:    [B, 1, H, W]  predicted DTM (gated, normalised)
        g_target: [B, 1, H, W]  GT DTM (normalised)
        logit:    [B, 1, H, W]  raw confidence logits ℓ from the UNet
        dsm:      [B, 1, H, W]  conditioning DSM (normalised)
        valid:    [B, H, W] or None — mask of pixels to include in loss
        m_alpha:  [B, 1, H, W] or [B, H, W] — precomputed ground mask
                  M_α = 1[|s_metres − g_metres| < α_metres]. If supplied,
                  used directly. If None, computed in NORMALISED space
                  using `alpha` (legacy fallback for unit tests).
        alpha:    legacy fallback — threshold in normalised units.
                  Only used if `m_alpha` is None. Paper §7.3 doesn't
                  specify a numerical value; with per-tile normalisation
                  this would be tile-dependent in metres, so prefer
                  precomputing m_alpha in metres at the dataset level
                  with a fixed Sithole-Vosselman 2003 α_metres = 0.20m.

    Returns:
        total_loss, dict of per-term values for logging.
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

    # Gradient magnitude L1: || |∇ĝ| − |∇g| ||₁
    grad_pred = _grad_magnitude(g_hat)
    grad_gt   = _grad_magnitude(g_target)
    l_grad = ((grad_pred - grad_gt).abs() * v).sum() / n_valid

    # Confidence supervision: BCE on σ(ℓ) against the M_α target
    if m_alpha is not None:
        ma = m_alpha.to(g_hat.dtype)
        if ma.dim() == 3:
            ma = ma.unsqueeze(1)
    else:
        # Legacy fallback: compute in normalised space (tile-dependent metres)
        with torch.no_grad():
            ma = ((dsm - g_target).abs() < alpha).to(g_hat.dtype)
    bce = F.binary_cross_entropy_with_logits(
        logit, ma, reduction='none')
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

"""Normalization (paper §7.2 Eq.15).

Joint min-max from valid pixels of BOTH DSM and DTM, mapping each to
[-1, 1]. Invalid regions are set to 0 (per paper).

Why joint and not per-tile separate? The paper's reasoning: the gating
G(r̂, ℓ, s) = σ(ℓ)⊙s + (1-σ(ℓ))⊙(s - r̂) requires DSM s and predicted
DTM ĝ to share a frame so that residual r = s - g is meaningful. Per-
raster mins/maxes would make r ill-defined.
"""
from __future__ import annotations
import numpy as np


def joint_minmax_normalise(dsm: np.ndarray, dtm: np.ndarray,
                            valid: np.ndarray):
    """Joint min-max scaling to [-1, 1] using paper Eq.15.

    Args:
        dsm:   [H, W] float32 DSM, NaN or any value outside `valid` is ignored
        dtm:   [H, W] float32 DTM target, same convention
        valid: [H, W] bool/float mask of pixels considered "valid" in both

    Returns:
        dsm_norm, dtm_norm, stats={'vmin', 'vmax', 'has_data'}
        Both normed arrays have invalid pixels set to 0 (paper convention).
    """
    valid = valid.astype(bool)
    if not valid.any():
        return (np.zeros_like(dsm, dtype=np.float32),
                np.zeros_like(dtm, dtype=np.float32),
                dict(vmin=0.0, vmax=1.0, has_data=False))

    s_valid = dsm[valid]
    g_valid = dtm[valid]
    # Robust to NaN if upstream forgot to mask them.
    s_valid = s_valid[np.isfinite(s_valid)]
    g_valid = g_valid[np.isfinite(g_valid)]
    if s_valid.size == 0 or g_valid.size == 0:
        return (np.zeros_like(dsm, dtype=np.float32),
                np.zeros_like(dtm, dtype=np.float32),
                dict(vmin=0.0, vmax=1.0, has_data=False))

    vmin = float(min(s_valid.min(), g_valid.min()))
    vmax = float(max(s_valid.max(), g_valid.max()))
    span = max(vmax - vmin, 1e-6)

    def _norm(x):
        out = np.zeros_like(x, dtype=np.float32)
        m = valid & np.isfinite(x)
        out[m] = 2.0 * (x[m].astype(np.float32) - vmin) / span - 1.0
        return out

    return _norm(dsm), _norm(dtm), dict(vmin=vmin, vmax=vmax, has_data=True)


def denormalise(x_norm: np.ndarray, stats: dict) -> np.ndarray:
    """Invert Eq.15 back to metres."""
    if not stats.get('has_data', False):
        return np.zeros_like(x_norm, dtype=np.float32)
    vmin = float(stats['vmin'])
    vmax = float(stats['vmax'])
    return ((x_norm.astype(np.float32) + 1.0) * 0.5 * (vmax - vmin) + vmin
            ).astype(np.float32)

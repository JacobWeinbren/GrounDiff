"""Per-tile min-max normalisation (GrounDiff supplement §7.2, adapted).

The paper maps each tile to [-1, 1] using the min and max over valid pixels of
BOTH the DSM and the ground-truth DTM, and sets invalid pixels to 0. The GT
DTM is not available at inference, so using it in training creates a
train/inference mismatch. Here the range comes from `norm_channels`: inputs
that exist at both training and inference time (by default dsm_max and
dsm_min, plus the lasground_new DTM in before -> after mode, which is a close
stand-in for g). This is a documented deviation.

`min_range` (metres) stops nearly flat tiles from being stretched so far that
unit-variance diffusion noise corresponds to a few centimetres; tiles with
less relief than this are centred inside a window of `min_range`.
"""
from __future__ import annotations

import numpy as np


def tile_range(stack: np.ndarray, valid: np.ndarray, min_range: float = 2.0) -> tuple[float, float]:
    """stack: [C, H, W] reference rasters in metres; valid: [H, W] bool.

    Returns (lo, scale) such that x_n = 2 (x - lo) / scale - 1.
    """
    if valid.any():
        vals = stack[:, valid]
        vals = vals[np.isfinite(vals)]
    else:
        vals = np.empty(0)
    if vals.size == 0:
        return 0.0, float(min_range)
    lo, hi = float(vals.min()), float(vals.max())
    scale = hi - lo
    if scale < min_range:
        mid = 0.5 * (lo + hi)
        lo, scale = mid - 0.5 * min_range, float(min_range)
    return lo, scale


def normalise(x, lo, scale):
    return 2.0 * (x - lo) / scale - 1.0


def denormalise(xn, lo, scale):
    return (xn + 1.0) * 0.5 * scale + lo


HEIGHT_CHANNELS = {"dsm_max", "dsm_min", "dsm_last", "dtm_before", "gt_dtm"}
NEAREST_CHANNELS = {"has_return", "sem_ground", "sem_nonground", "gt_valid", "before_valid"}


def channel_transform(name: str, x: np.ndarray, lo: float, scale: float) -> np.ndarray:
    """Map a raster channel (metres or counts) into network units."""
    if name in HEIGHT_CHANNELS:
        return normalise(x, lo, scale)
    if name == "z_std":
        return 2.0 * x / scale
    if name == "density":
        return np.log1p(np.maximum(x, 0.0)) / 4.0
    if name == "echoes":
        return (x - 1.0) / 2.0
    return x

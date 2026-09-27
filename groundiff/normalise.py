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


def tile_range(stack: np.ndarray, valid: np.ndarray | None = None, min_range: float = 2.0,
               quantile: float = 0.0) -> tuple[float, float]:
    """stack: [C, H, W] reference rasters in metres. The range covers every
    finite value of every channel (valid=None) or only pixels where `valid`
    is True. quantile > 0 uses the [q, 1-q] quantiles instead of min/max, so a
    single stray noise return cannot stretch the tile.

    Returns (lo, scale) such that x_n = 2 (x - lo) / scale - 1.
    """
    vals = stack if valid is None else stack[:, valid]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, float(min_range)
    if quantile > 0:
        lo, hi = (float(v) for v in np.quantile(vals, [quantile, 1.0 - quantile]))
    else:
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
NEAREST_CHANNELS = {"has_return", "sem_ground", "sem_nonground", "gt_valid", "before_valid", "top_ground",
                    "in_survey", "flat_water"}
FILL_CHANNELS = {"dsm_max", "dsm_min", "dsm_last", "dtm_before"}
# Channels computed per tile rather than read from a raster. tile_range tells the network the
# tile's height range in metres (constant over the tile), which the [-1, 1] normalisation hides:
# log(scale / 10 m) / 2, i.e. -0.8 for a flat tile (the 2 m minimum range), +1.2 for 500 m of relief.
# The target stays normalised as in GrounDiff; absolute height is never given (no geography).
VIRTUAL_CHANNELS = {"tile_range"}


def virtual_channel(name: str, shape: tuple, lo: float, scale: float) -> np.ndarray:
    if name == "tile_range":
        return np.full(shape, 0.5 * np.log(max(float(scale), 1e-3) / 10.0), np.float32)
    raise KeyError(name)


def fill_nearest(a: np.ndarray) -> np.ndarray:
    """Replace NaNs by the nearest valid value (no-op if none are valid)."""
    bad = ~np.isfinite(a)
    if not bad.any() or bad.all():
        return a
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


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


def coverage_mask(has_data: np.ndarray, gsd: float, close_m: float = 30.0) -> np.ndarray:
    """Cells inside LiDAR coverage: data cells plus voids narrower than
    2 * close_m (morphological closing; e.g. rivers, small ponds, shadows).
    Local by construction, so separately processed neighbouring blocks agree
    (hole filling would depend on each block's extent). Larger voids (big
    lakes, sea, outside the survey) stay outside."""
    from scipy.ndimage import distance_transform_edt
    m = np.asarray(has_data, bool)
    if not m.any():
        return m
    r = close_m / gsd
    if r > 0:
        dilated = distance_transform_edt(~m) <= r
        m = m | (distance_transform_edt(dilated) > r)
    return m

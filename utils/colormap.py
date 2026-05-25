"""Colourblind-safe geographic palettes for ALS rasters.

The palettes here are inspired by Fabio Crameri's "Scientific Colour Maps"
(https://www.fabiocrameri.ch/colourmaps/), which are perceptually uniform,
colourblind-safe (deuteranopia, protanopia, tritanopia), AND render
sensibly when printed in greyscale. They look "earthy" / professional
instead of the rainbow green→red that's typical of GIS software but
that fails for ~8 % of male viewers.

Three palettes:

  hypsometric  -- sequential, dark teal → green-blue → olive → ochre →
                  warm tan → cream. Inspired by Crameri's `bukavu` and
                  `batlow`. Reads as "topographic" but avoids the
                  green/red conflict.

  bwr          -- diverging, deep blue → near-white → deep red, with a
                  tight neutral band at zero. Inspired by Crameri's
                  `vik`, which is the canonical colourblind-safe
                  diverging map. Distinguishable luminance on each
                  side so it works in greyscale too.

  classification -- two muted, high-contrast colours for binary
                    ground / non-ground panels. We use a slate-teal /
                    warm-ochre pair (similar luminance) — the
                    blue-orange axis is the safest binary distinction
                    across all colourblindness types.

`fill_sentinels()` and `hillshade()` / `shade_blend()` provide the
display-cosmetics layer used by the visualisation script.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ---- Palette definitions ----------------------------------------------- #

# Topographic palette inspired by classical hypsometric maps (see e.g. the
# GrounDiff RT dataset visualisation) but adjusted for colourblind safety
# via LUMINANCE bumping: low and high ends are both dark; the bright band
# sits in the middle (golden yellow). This means a deuteranopic / protanopic
# viewer who can't separate green vs red can still read the map because the
# *brightness* is monotonic from end → mid → other end (V-shape).
#
# Result: looks like a classical green-yellow-orange-brown DTM but stays
# readable for the ~8 % of male viewers with red-green colour deficiency
# and prints sensibly in greyscale.
_HYPSO_STOPS = np.array([
    # (t in [0, 1], R, G, B in [0, 1])
    (0.00, 0.047, 0.271, 0.247),   # very dark teal-green (water / lowest)
    (0.10, 0.176, 0.529, 0.325),   # forest green
    (0.25, 0.404, 0.753, 0.302),   # bright spring green
    (0.42, 0.737, 0.831, 0.294),   # yellow-green
    (0.55, 0.929, 0.788, 0.275),   # warm gold  (peak luminance)
    (0.70, 0.871, 0.553, 0.235),   # orange-tan
    (0.83, 0.624, 0.349, 0.196),   # terracotta brown
    (0.94, 0.353, 0.196, 0.118),   # deep brown
    (1.00, 0.137, 0.071, 0.039),   # near-black brown (peaks)
], dtype=np.float64)


# Diverging blue-white-red. Inspired by Crameri's `vik`.
# Luminance is symmetric around the white centre so colourblind viewers
# can still tell direction by darkness; the hue separation is on the
# blue-orange axis (safe across deuteranopia/protanopia/tritanopia).
_BWR_STOPS = np.array([
    (0.00, 0.020, 0.188, 0.380),   # deep blue
    (0.20, 0.137, 0.408, 0.620),   # mid blue
    (0.40, 0.553, 0.741, 0.851),   # light blue
    (0.48, 0.918, 0.949, 0.973),   # near-white
    (0.50, 1.000, 1.000, 1.000),   # white at zero
    (0.52, 0.973, 0.949, 0.918),   # near-white
    (0.60, 0.965, 0.741, 0.553),   # light orange
    (0.80, 0.776, 0.376, 0.176),   # mid orange-red
    (1.00, 0.475, 0.090, 0.012),   # deep red-brown
], dtype=np.float64)


# Classification: muted slate-teal (ground) vs warm ochre (non-ground).
# Blue-orange is the universal safe pair across all colourblindness types.
GROUND_RGB = (0.349, 0.510, 0.580)       # muted slate-teal
NON_GROUND_RGB = (0.851, 0.616, 0.314)   # warm ochre

# Greys for masked / invalid / unreliable regions. Lighter than before so
# they read as "no data here" without dominating the panel.
INVALID_RGB = (0.847, 0.847, 0.847)      # light grey
HATCH_RGB = (0.643, 0.643, 0.643)        # medium grey for hatch lines


# ---- Colormap application ---------------------------------------------- #

def _interp_cmap(stops: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation through palette stops."""
    t = np.clip(t, 0.0, 1.0)
    out = np.empty(t.shape + (3,), dtype=np.float32)
    ts = stops[:, 0]
    cs = stops[:, 1:]
    for ch in range(3):
        out[..., ch] = np.interp(t, ts, cs[:, ch])
    return out


def hypsometric(z: np.ndarray, *, vmin: Optional[float] = None,
                 vmax: Optional[float] = None,
                 valid: Optional[np.ndarray] = None,
                 invalid_rgb=INVALID_RGB) -> np.ndarray:
    """Colourblind-safe sequential elevation map -> (H, W, 3) float32."""
    if valid is None:
        v = np.ones_like(z, dtype=bool)
    else:
        v = valid.astype(bool)
    if vmin is None or vmax is None:
        if v.any():
            zv = z[v]
            vmin_, vmax_ = np.quantile(zv, [0.02, 0.98])
        else:
            vmin_, vmax_ = float(z.min()), float(z.max())
        if vmin is None: vmin = float(vmin_)
        if vmax is None: vmax = float(vmax_)
    span = max(vmax - vmin, 1e-9)
    t = (z - vmin) / span
    rgb = _interp_cmap(_HYPSO_STOPS, t)
    if not v.all():
        rgb[~v] = np.array(invalid_rgb, dtype=np.float32)
    return rgb


def diverging_residual(r: np.ndarray, *, vrange: float = 1.0,
                        valid: Optional[np.ndarray] = None,
                        invalid_rgb=INVALID_RGB) -> np.ndarray:
    """Colourblind-safe diverging error map, white at zero. vrange in m."""
    if valid is None:
        v = np.ones_like(r, dtype=bool)
    else:
        v = valid.astype(bool)
    t = np.clip((r + vrange) / (2 * vrange), 0.0, 1.0)
    rgb = _interp_cmap(_BWR_STOPS, t)
    if not v.all():
        rgb[~v] = np.array(invalid_rgb, dtype=np.float32)
    return rgb


def classification_bicolor(pred_ground: np.ndarray,
                            valid: Optional[np.ndarray] = None,
                            *, ground_rgb=GROUND_RGB,
                            ng_rgb=NON_GROUND_RGB,
                            invalid_rgb=INVALID_RGB) -> np.ndarray:
    """Binary classification map: slate-teal ground vs warm-ochre non-ground.

    Blue-orange pair is colourblind-safe across all common types.
    """
    pred_ground = pred_ground.astype(bool)
    rgb = np.empty(pred_ground.shape + (3,), dtype=np.float32)
    rgb[pred_ground] = np.array(ground_rgb, dtype=np.float32)
    rgb[~pred_ground] = np.array(ng_rgb, dtype=np.float32)
    if valid is not None:
        rgb[~valid.astype(bool)] = np.array(invalid_rgb, dtype=np.float32)
    return rgb


def hatched_overlay(rgb: np.ndarray, mask: np.ndarray,
                     *, line_spacing: int = 7,
                     line_color=HATCH_RGB) -> np.ndarray:
    """Subtle diagonal hatching to flag "unreliable" regions
    (e.g. cells where GT was interpolated)."""
    if not mask.any():
        return rgb
    out = rgb.copy()
    ys, xs = np.where(mask)
    keep = (((xs + ys) % line_spacing) == 0)
    if keep.any():
        out[ys[keep], xs[keep]] = np.array(line_color, dtype=np.float32)
    return out


# ---- Display helpers --------------------------------------------------- #

def fill_sentinels(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Display-only: fill empty cells (mask=False) with their nearest
    valid neighbour's value so panels look continuous."""
    if mask.all():
        return z.copy()
    if not mask.any():
        return np.zeros_like(z)
    from scipy.ndimage import distance_transform_edt
    _, ind = distance_transform_edt(~mask.astype(bool),
                                     return_distances=True,
                                     return_indices=True)
    return z[tuple(ind)]


def hillshade(elev: np.ndarray, *, azimuth_deg: float = 315.0,
               altitude_deg: float = 45.0, gsd: float = 1.0,
               z_factor: float = 1.0) -> np.ndarray:
    """Lambertian hillshade in [0, 1] (315° = NW light, standard)."""
    z = elev.astype(np.float64) * z_factor
    dy, dx = np.gradient(z, gsd, gsd)
    norm = np.sqrt(dx * dx + dy * dy + 1.0)
    nx, ny, nz = -dx / norm, -dy / norm, 1.0 / norm
    az = np.deg2rad(azimuth_deg)
    al = np.deg2rad(altitude_deg)
    lx = np.cos(al) * np.sin(az)
    ly = np.cos(al) * np.cos(az)
    lz = np.sin(al)
    shade = nx * lx + ny * ly + nz * lz
    return np.clip(shade, 0.0, 1.0).astype(np.float32)


def shade_blend(rgb: np.ndarray, shade: np.ndarray, *,
                 strength: float = 0.50) -> np.ndarray:
    """Multiplicative shading. strength=0.5 keeps colour saturated."""
    factor = (1.0 - strength + strength * shade)[..., None].astype(np.float32)
    return np.clip(rgb * factor, 0.0, 1.0)

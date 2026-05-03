"""GrounDiff §7.1 augmentation.

Pipeline, each step applied with probability 0.5 (paper):
  1. Random rotation from {0°, 90°, 180°, 270°} + jitter in (-5°, 5°)
  2. Multi-scale resize to one of {256, 512, 1024}
  3. Random crop to 256×256
  4. Horizontal flip
  5. Vertical flip

Final output: always 256×256 (DSM, DTM, valid_mask).

The DSM and DTM share one frame (paper §7.2), so the same random crop /
resize / flip / rotation parameters apply to both — implemented by
processing them as a stacked tensor.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np


def _maybe_rotate(stack, valid, rng, p=0.5):
    """Rotate by k*90° (k uniform in {0,1,2,3}) plus a small jitter in
    (-5°, 5°). Step applied with probability p."""
    if rng.random() >= p:
        return stack, valid
    k = int(rng.integers(0, 4))
    if k:
        stack = torch.rot90(stack, k, dims=(-2, -1))
        valid = torch.rot90(valid, k, dims=(-2, -1))
    # Jitter: small affine rotation. Use grid_sample with bilinear for
    # cont. data and nearest for the mask. Only apply if jitter > 0.5°
    # to avoid wasting compute on sub-degree motion.
    deg = float(rng.uniform(-5.0, 5.0))
    if abs(deg) < 0.5:
        return stack, valid
    rad = float(np.deg2rad(deg))
    cos, sin = float(np.cos(rad)), float(np.sin(rad))
    H, W = stack.shape[-2:]
    theta = torch.tensor(
        [[[cos, -sin, 0.0], [sin, cos, 0.0]]],
        dtype=stack.dtype, device=stack.device,
    )
    grid = F.affine_grid(theta, [1, stack.shape[0], H, W],
                          align_corners=False)
    stack = F.grid_sample(stack.unsqueeze(0), grid, mode='bilinear',
                           padding_mode='zeros', align_corners=False)[0]
    valid = F.grid_sample(valid.unsqueeze(0).unsqueeze(0), grid,
                           mode='nearest', padding_mode='zeros',
                           align_corners=False)[0, 0]
    return stack, valid


def _maybe_resize(stack, valid, rng, sizes=(256, 512, 1024), p=0.5):
    """Resize to a randomly-picked size from `sizes`. Skipped with
    prob (1-p). Bilinear for continuous channels, nearest for the
    valid mask to preserve crisp boundaries."""
    if rng.random() >= p:
        return stack, valid
    new_size = int(rng.choice(sizes))
    if stack.shape[-1] == new_size and stack.shape[-2] == new_size:
        return stack, valid
    stack = F.interpolate(stack.unsqueeze(0), size=(new_size, new_size),
                           mode='bilinear', align_corners=False)[0]
    valid = F.interpolate(valid.unsqueeze(0).unsqueeze(0).float(),
                           size=(new_size, new_size),
                           mode='nearest')[0, 0]
    return stack, valid


def _crop_to(stack, valid, crop, rng):
    """Random crop to `crop`x`crop`. Pads (replicate) if input is
    smaller than crop along any dim — protects against tiny tiles
    coming out of preprocess for narrow LAZ files."""
    H, W = stack.shape[-2:]
    pad_h = max(0, crop - H)
    pad_w = max(0, crop - W)
    if pad_h or pad_w:
        stack = F.pad(stack, (0, pad_w, 0, pad_h), mode='replicate')
        valid = F.pad(valid.unsqueeze(0).unsqueeze(0).float(),
                       (0, pad_w, 0, pad_h),
                       mode='constant', value=0.0)[0, 0]
        H, W = stack.shape[-2:]
    if H == crop and W == crop:
        return stack, valid
    i = int(rng.integers(0, H - crop + 1))
    j = int(rng.integers(0, W - crop + 1))
    return (stack[..., i:i + crop, j:j + crop],
            valid[..., i:i + crop, j:j + crop])


def _maybe_flip(stack, valid, rng, p=0.5):
    """Horizontal flip with prob p, vertical flip with prob p
    (independent). Paper says both flips, each at 0.5."""
    if rng.random() < p:
        stack = torch.flip(stack, dims=(-1,))
        valid = torch.flip(valid, dims=(-1,))
    if rng.random() < p:
        stack = torch.flip(stack, dims=(-2,))
        valid = torch.flip(valid, dims=(-2,))
    return stack, valid


def groundiff_augment(dsm: torch.Tensor, dtm: torch.Tensor,
                      valid: torch.Tensor, *,
                      crop: int = 256,
                      rng: np.random.Generator | None = None,
                      p: float = 0.5,
                      resize_choices=(256, 512, 1024),
                      extra: torch.Tensor | None = None):
    """Apply paper §7.1 augmentation pipeline.

    Args:
        dsm:   [1, H, W]  float32 in [-1, 1] (already normalised)
        dtm:   [1, H, W]  float32 in [-1, 1]
        valid: [H, W]     float32 in {0, 1}
        crop:  output spatial size
        rng:   numpy RNG; if None a new one is made
        p:     per-step apply probability
        resize_choices: candidate output sizes for the resize step
        extra: [C, H, W] optional extra continuous channels to augment
               jointly with DSM/DTM (e.g. dsm_min). Same geometric ops
               applied identically.

    Returns:
        If extra is None: (dsm_out, dtm_out, valid_out)
        Else: (dsm_out, dtm_out, valid_out, extra_out)
        All spatial dim = (crop, crop).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Stack so all geometric ops apply identically across rasters
    if extra is None:
        stack = torch.cat([dsm, dtm], dim=0)               # [2, H, W]
        n_dsm, n_dtm, n_extra = 1, 1, 0
    else:
        stack = torch.cat([dsm, dtm, extra], dim=0)        # [2+C, H, W]
        n_dsm, n_dtm, n_extra = 1, 1, extra.shape[0]

    stack, valid = _maybe_rotate(stack, valid, rng, p)
    stack, valid = _maybe_resize(stack, valid, rng,
                                  sizes=resize_choices, p=p)
    stack, valid = _crop_to(stack, valid, crop, rng)
    stack, valid = _maybe_flip(stack, valid, rng, p)

    # Defensive: ensure final size is exactly (crop, crop)
    if stack.shape[-1] != crop or stack.shape[-2] != crop:
        stack = F.interpolate(stack.unsqueeze(0), size=(crop, crop),
                               mode='bilinear', align_corners=False)[0]
        valid = F.interpolate(valid.unsqueeze(0).unsqueeze(0).float(),
                               size=(crop, crop), mode='nearest')[0, 0]

    if n_extra == 0:
        return stack[0:1], stack[1:2], valid
    return (stack[0:1], stack[1:2], valid,
            stack[2:2 + n_extra])

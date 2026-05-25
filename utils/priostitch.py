"""PrioStitch: paper §3.4 — two-pass inference for scenes larger than one tile.

Pass 1 (coarse):
    Downsample the whole scene's 4-channel raster to a tractable size
    (e.g. 1024 × 1024), run one forward pass of the diffusion through
    `init='noisy_dsm'`. This gives a global DTM prior at low resolution.

Pass 2 (fine):
    Upsample the coarse prior back to the scene's native resolution.
    Then for each fine-resolution tile, run the diffusion with
    `init='priostitch'` and `prior_dtm=upsampled_coarse[tile]`. The
    PrioStitch init is `q_sample(prior_dtm, γ_T)` rather than
    `q_sample(dsm, γ_T)`, so the network refines a tile-local view of
    the global prior.

Tile stitching:
    We tile with overlap and average overlapping predictions via a
    raised-cosine window to avoid seam artefacts. Each tile is
    normalised independently (per-tile [-1, +1] using its own dsm_max
    range), so we de-normalise back to metres before averaging.

Input:
    The same raster.npz format as preprocess.py emits. We work on the
    arrays directly (not LAZ).

Output:
    `dtm_pred` (H, W) float32 — predicted DTM in absolute metres.
    `logit`    (H, W) float32 — final-step confidence logit ℓ.
    `prob_ground` (H, W) float32 — σ(ℓ).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _tta_transforms(n: int):
    """Per-tile test-time augmentation transforms.

    Ported from the stage 1 image-GrounDiff codebase. Each entry is
    (forward, inverse): forward applies a flip/rotation to the input,
    inverse undoes it on the prediction so we can average tiles in the
    canonical orientation.

    n=1  -> identity only (no TTA).
    n=4  -> D2 dihedral group: identity + h-flip + v-flip + 180° rot.
    n=8  -> full D4 group: D2 + three 90° rotations + one diagonal flip.

    Cost scales linearly: 4× and 8× the per-tile inference time
    respectively. With our tile=1024 and T=10 that's ~5-10s extra per
    tile at TTA=8 — fine for the final test/inference pass but
    prohibitive for the per-val viz hook, so val viz keeps tta=1.
    """
    I = lambda t: t
    hflip = lambda t: torch.flip(t, dims=[-1])
    vflip = lambda t: torch.flip(t, dims=[-2])
    hvflip = lambda t: torch.flip(t, dims=[-1, -2])
    rot1 = lambda t: torch.rot90(t, 1, dims=[-2, -1])
    rot2 = lambda t: torch.rot90(t, 2, dims=[-2, -1])
    rot3 = lambda t: torch.rot90(t, 3, dims=[-2, -1])
    rot1_inv = lambda t: torch.rot90(t, -1, dims=[-2, -1])
    rot3_inv = lambda t: torch.rot90(t, -3, dims=[-2, -1])
    if n <= 1:
        return [(I, I)]
    if n <= 4:
        return [(I, I), (hflip, hflip), (vflip, vflip), (hvflip, hvflip)]
    # n=8 -> full D4
    return [
        (I, I), (hflip, hflip), (vflip, vflip), (hvflip, hvflip),
        (rot1, rot1_inv), (rot2, rot2), (rot3, rot3_inv),
        # one extra: rot1 ∘ hflip → diagonal reflection
        (lambda t: torch.rot90(torch.flip(t, [-1]), 1, [-2, -1]),
         lambda t: torch.flip(torch.rot90(t, -1, [-2, -1]), [-1])),
    ]


def _raised_cosine_window(tile_size: int, edge_frac: float = 0.25,
                           floor: float = 0.05) -> torch.Tensor:
    """A 2D raised-cosine window that ramps from `floor`→1 over the
    outer `edge_frac` of each side. Used to weight overlapping tile
    predictions so seams blend smoothly.

    The floor (default 0.05) ensures the window is never exactly zero,
    so even at scene corners where a single tile covers the pixel the
    accumulator gets meaningful weight and `sum_dtm / sum_w` produces a
    sensible value instead of zero-divided-by-tiny.
    """
    n = tile_size
    edge = max(1, int(n * edge_frac))
    w = torch.ones(n, dtype=torch.float32)
    if edge > 1:
        ramp = 0.5 * (1 - torch.cos(torch.linspace(0, np.pi, edge)))
        w[:edge] = ramp
        w[-edge:] = ramp.flip(0)
    w = floor + (1.0 - floor) * w
    return w[:, None] * w[None, :]  # (n, n)


def _linear_window(tile_size: int, floor: float = 1e-3) -> torch.Tensor:
    """2D linear window: weight = normalised distance from nearest edge.

    Centre weight 1.0, edge weight 0 (clamped to `floor`). Matches paper
    §12.2 "linear weighting based on distance from tile edge" and the
    official reference's _make_blend_weights('linear').
    """
    n = tile_size
    coord = (torch.arange(n, dtype=torch.float32) + 0.5) / n
    d = torch.minimum(coord, 1.0 - coord) * 2.0   # 0 at edges, 1 at centre
    w_2d = torch.minimum(d[:, None], d[None, :])
    return torch.clamp(w_2d, min=floor)


def _exp_window(tile_size: int) -> torch.Tensor:
    """2D exponential decay window: weight = exp(-2 · (1 - d_edge_norm)).

    Centre weight 1.0, edge weight exp(-2) ≈ 0.135. Matches paper
    §12.2 "exponential decay weighting" and the official reference's
    _make_blend_weights('exponential').
    """
    n = tile_size
    coord = (torch.arange(n, dtype=torch.float32) + 0.5) / n
    d = torch.minimum(coord, 1.0 - coord) * 2.0
    d_2d = torch.minimum(d[:, None], d[None, :])
    return torch.exp(-2.0 * (1.0 - d_2d)).clamp(min=1e-3)


def _to_device_inputs(*arrays, device):
    out = []
    for a in arrays:
        t = torch.from_numpy(a).float().to(device)
        if t.ndim == 2:
            t = t[None, None]
        elif t.ndim == 3:
            t = t[None]
        out.append(t)
    return out


def _normalise_pair(dsm_max: np.ndarray, dsm_min: np.ndarray,
                     dsm_mean: np.ndarray, dsm_mask: np.ndarray
                     ) -> tuple:
    """Inference-time normalisation. MUST match the training frame exactly
    (data/dataset.py::_tile_normalise) or the model sees a shifted input
    distribution at test time. Both bounds come from the max-z DSM over
    `dsm_mask`; dsm_min/dsm_mean are normalised into that frame and
    clamped to [-1, 1] (dsm_min can fall below the max-z minimum).

    Returns (max_n, min_n, mean_n, z_lo, z_hi).
    """
    real = dsm_mask.astype(bool)
    if not real.any():
        z_lo, z_hi = 0.0, 1.0
    else:
        z_lo = float(dsm_max[real].min())
        z_hi = float(dsm_max[real].max())
    span = max(z_hi - z_lo, 1e-3)
    def n(a): return ((a - z_lo) / span * 2.0 - 1.0).astype(np.float32)
    # Paper §7.2 / Eq. 15: normalise then set invalid regions to 0,
    # matching data/dataset.py::_tile_normalise exactly. Inference is not
    # augmented, so no anti-bleed fill is needed here — only the no-data
    # value matters, and it must equal the training value (0).
    max_n = np.clip(n(dsm_max), -1.0, 1.0).astype(np.float32)
    min_n = np.clip(n(dsm_min), -1.0, 1.0).astype(np.float32)
    mean_n = np.clip(n(dsm_mean), -1.0, 1.0).astype(np.float32)
    max_n[~real] = 0.0
    min_n[~real] = 0.0
    mean_n[~real] = 0.0
    return max_n, min_n, mean_n, z_lo, z_hi


@torch.no_grad()
def coarse_pass(
    model,
    dsm_max: np.ndarray,
    dsm_min: np.ndarray,
    dsm_mean: np.ndarray,
    dsm_mask: np.ndarray,
    *,
    coarse_size: int = 256,
    device: str | torch.device = 'cuda',
) -> tuple[np.ndarray, float, float]:
    """Run one forward pass on a downsampled view of the whole scene.

    Returns (coarse_dtm_metres, z_lo, z_hi). The downsample factor is
    chosen so the longer side matches `coarse_size` (rounded to a
    multiple of 16 = 2 * patch_size = 16 to satisfy DiT's divisibility).
    """
    device = torch.device(device)
    H, W = dsm_max.shape
    scale = coarse_size / max(H, W)
    # Round target shape down to a multiple of the model's spatial
    # divisibility. For our UNet this is 2 ** num_downsamples (e.g.
    # 4 levels → 3 downsamples → multiple of 8). Falls back to 16 if
    # the model exposes neither attribute (legacy DiT path).
    if hasattr(model.dit, 'grid_multiple'):
        block = int(model.dit.grid_multiple)
    elif hasattr(model.dit, 'patch_size'):
        block = 2 * int(model.dit.patch_size)
    else:
        block = 16
    new_H = max(block, int(round(H * scale / block)) * block)
    new_W = max(block, int(round(W * scale / block)) * block)

    dm_n, dn_n, dmean_n, z_lo, z_hi = _normalise_pair(
        dsm_max, dsm_min, dsm_mean, dsm_mask)

    # Downsample to (new_H, new_W). For max/min we use max/min pooling
    # (preserving extrema is important). For mean we use avg pooling.
    # For mask: any cell in the block having data → mask=1.
    def _down_max(arr, mode):
        t = torch.from_numpy(arr)[None, None].float()
        if mode == 'max':
            return F.adaptive_max_pool2d(t, (new_H, new_W))[0, 0].numpy()
        elif mode == 'min':
            return -F.adaptive_max_pool2d(-t, (new_H, new_W))[0, 0].numpy()
        elif mode == 'mean':
            return F.adaptive_avg_pool2d(t, (new_H, new_W))[0, 0].numpy()
        else:
            raise ValueError(mode)

    dm_c = _down_max(dm_n, 'max')
    dn_c = _down_max(dn_n, 'min')
    dmean_c = _down_max(dmean_n, 'mean')
    # Mask: average then threshold; anything with positive mean has data.
    mask_c = (_down_max(dsm_mask.astype(np.float32), 'max') > 0.5).astype(np.float32)

    # To tensors.
    dm_t, dn_t, dmean_t, mask_t = _to_device_inputs(
        dm_c, dn_c, dmean_c, mask_c, device=device)

    # Run reverse diffusion with noisy_dsm init.
    dtm_pred = model.infer(
        dsm_max=dm_t, dsm_min=dn_t, dsm_mean=dmean_t, dsm_mask=mask_t,
        init='noisy_dsm')[0, 0]                              # [Hc, Wc]
    # De-normalise back to metres.
    dtm_pred_m = ((dtm_pred.float().cpu().numpy() + 1.0) / 2.0 * (z_hi - z_lo)
                   + z_lo).astype(np.float32)
    return dtm_pred_m, z_lo, z_hi


@torch.no_grad()
def fine_pass(
    model,
    dsm_max: np.ndarray,
    dsm_min: np.ndarray,
    dsm_mean: np.ndarray,
    dsm_mask: np.ndarray,
    coarse_dtm_m: np.ndarray | None,
    *,
    tile_size: int = 256,
    overlap: int = 128,
    blend_mode: str = 'linear',
    device: str | torch.device = 'cuda',
    use_priostitch: bool = True,
    tta: int = 1,
    progress_cb=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tile-by-tile inference at native raster resolution.

    Args:
        coarse_dtm_m: [Hc, Wc] coarse DTM from `coarse_pass`, in METRES
                      (absolute). Upsampled to (H, W) and used as the
                      PrioStitch init prior. Pass None to use 'noisy_dsm'
                      init (paper baseline, no PrioStitch).
        tile_size:   tile side in pixels (must divide model.dit.grid_multiple).
                      Default 256 = paper-faithful.
        overlap:     pixels of overlap between adjacent tiles.
                      Default 128 = paper §8 Table 7: "tiles overlap by
                      50 percent, stride 128" (at tile_size=256).
        blend_mode:  How to merge overlapping tile DTM predictions.
                      Paper §12.2 / Table 7 lists six modes; all
                      supported here:

                      'min'        : pixel-wise minimum across overlapping
                                     tiles. Paper Tab.7(e) — best regression
                                     RMSE on DALES. Default.
                      'mean'       : simple arithmetic average across tiles.
                                     Paper Tab.7(d).
                      'max'        : pixel-wise maximum. Paper Tab.7(f),
                                     rarely wins (preserves above-ground
                                     artifacts).
                      'linear'     : windowed weighted mean with linear
                                     edge-to-centre ramp. Paper Tab.7(g) —
                                     "best balance of performance and
                                     visual quality".
                      'cosine'     : raised-cosine windowed weighted mean.
                                     Paper Tab.7(h).
                      'exponential': exp-decay windowed weighted mean
                                     (centre weight 1, edge ≈ 0.135).
                                     Paper Tab.7(i).

                      Logits ALWAYS use cosine-windowed mean regardless
                      of `blend_mode` — min/max on logits doesn't have a
                      sensible interpretation, and σ(ℓ) smoothness across
                      seams is desirable.
        tta:         1 | 4 | 8 — test-time augmentation passes per tile.
                     Each tile is run under N D2/D4 symmetry transforms
                     and averaged in canonical orientation. Costs `tta`×
                     inference time but reduces aliasing and improves
                     robustness near tile edges. Default 1 (off).

    Returns:
        dtm_pred_m: [H, W] predicted DTM in metres.
        logit:      [H, W] final-step logit ℓ.
    """
    if blend_mode not in ('min', 'max', 'mean', 'cosine', 'linear', 'exponential'):
        raise ValueError(
            f"blend_mode={blend_mode!r} not supported. Choose from "
            "'min', 'max', 'mean', 'cosine', 'linear', 'exponential'.")
    device = torch.device(device)
    H, W = dsm_max.shape
    # If the whole scene is smaller than one tile in either axis, pad the
    # arrays up to tile_size BEFORE allocating accumulators / upsampling the
    # prior (everything below sizes off H, W). Reflect elevation so the
    # border isn't an artificial cliff; zero the mask so padded cells are
    # no-data. We crop back to (H_orig, W_orig) before returning. Without
    # this, a scene narrower than tile_size in one axis produced a single
    # origin at 0 whose tile couldn't be tile_size wide -> the assertion
    # "tile_size doesn't fit" (the smoke-test crash). DEFRA tiles vary in
    # shape so some are < 256 in one dimension.
    H_orig, W_orig = H, W
    pad_h = max(0, tile_size - H)
    pad_w = max(0, tile_size - W)
    if pad_h or pad_w:
        def _pad(a, mode, **kw):
            return np.pad(a, ((0, pad_h), (0, pad_w)), mode=mode, **kw)
        dsm_max = (_pad(dsm_max, 'reflect') if (H > 1 and W > 1)
                   else _pad(dsm_max, 'edge'))
        dsm_min = _pad(dsm_min, 'edge')
        dsm_mean = _pad(dsm_mean, 'edge')
        dsm_mask = _pad(dsm_mask, 'constant', constant_values=0)
        H, W = dsm_max.shape
    if hasattr(model.dit, 'grid_multiple'):
        block = int(model.dit.grid_multiple)
    elif hasattr(model.dit, 'patch_size'):
        block = 2 * int(model.dit.patch_size)
    else:
        block = 16
    assert tile_size % block == 0, \
        f"tile_size={tile_size} must divide {block}"
    stride = tile_size - overlap

    # Upsample coarse prior to (H, W).
    prior_full = None
    if coarse_dtm_m is not None and use_priostitch:
        prior_full = F.interpolate(
            torch.from_numpy(coarse_dtm_m)[None, None].float(),
            size=(H, W), mode='bilinear', align_corners=False)[0, 0].numpy()

    # Output accumulators.
    # `sum_logit` always uses raised-cosine windowed mean for logits
    # (smooth σ(ℓ) across seams). DTM accumulator depends on blend_mode:
    #   'mean':         sum_dtm + counts, dtm_full = sum_dtm / count
    #   'min':          per-pixel minimum across tiles
    #   'max':          per-pixel maximum across tiles
    #   'cosine':       raised-cosine windowed weighted mean (current default)
    #   'linear':       linear-from-edge windowed weighted mean (paper Tab.7g)
    #   'exponential':  exp-decay windowed weighted mean (paper Tab.7i)
    sum_logit = np.zeros((H, W), dtype=np.float64)
    sum_w = np.zeros((H, W), dtype=np.float64)
    weighted_modes = ('cosine', 'linear', 'exponential')
    if blend_mode in weighted_modes:
        sum_dtm = np.zeros((H, W), dtype=np.float64)
    elif blend_mode == 'mean':
        sum_dtm = np.zeros((H, W), dtype=np.float64)
        cnt_dtm = np.zeros((H, W), dtype=np.int32)
    elif blend_mode == 'min':
        dtm_blend = np.full((H, W), np.inf, dtype=np.float64)
    elif blend_mode == 'max':
        dtm_blend = np.full((H, W), -np.inf, dtype=np.float64)

    # Logits always blend with the cosine window for smoothness.
    logit_window = _raised_cosine_window(tile_size).numpy()
    # DTM window depends on blend_mode (only used for weighted modes).
    if blend_mode == 'cosine':
        window = _raised_cosine_window(tile_size).numpy()
    elif blend_mode == 'linear':
        window = _linear_window(tile_size).numpy()
    elif blend_mode == 'exponential':
        window = _exp_window(tile_size).numpy()
    else:
        window = None  # min / max / mean — no DTM weighting kernel

    # Tile origins. Make sure to cover the right/bottom edge with one
    # final tile snapped to the boundary.
    def _origins(total, ts, st):
        os_ = list(range(0, total - ts + 1, st))
        if not os_:
            os_ = [0]
        elif os_[-1] + ts < total:
            os_.append(total - ts)
        return os_

    ys = _origins(H, tile_size, stride)
    xs = _origins(W, tile_size, stride)
    n_tiles = len(ys) * len(xs)
    t_done = 0

    for iy in ys:
        for ix in xs:
            t_done += 1
            iy_e = iy + tile_size
            ix_e = ix + tile_size
            assert iy_e <= H and ix_e <= W, \
                (f"tile out of bounds after padding "
                 f"(iy_e={iy_e}, ix_e={ix_e}, H={H}, W={W}); "
                 f"check overlap/origin logic")

            dm = dsm_max[iy:iy_e, ix:ix_e]
            dn = dsm_min[iy:iy_e, ix:ix_e]
            dme = dsm_mean[iy:iy_e, ix:ix_e]
            ma = dsm_mask[iy:iy_e, ix:ix_e]

            # Per-tile normalisation
            dm_n, dn_n, dme_n, z_lo, z_hi = _normalise_pair(dm, dn, dme, ma)
            dm_t, dn_t, dme_t, ma_t = _to_device_inputs(
                dm_n, dn_n, dme_n, ma.astype(np.float32), device=device)

            prior_t = None
            if prior_full is not None:
                prior_tile = prior_full[iy:iy_e, ix:ix_e]
                prior_n = ((prior_tile - z_lo) / max(z_hi - z_lo, 1e-3) * 2.0 - 1.0
                            ).astype(np.float32)
                prior_t, = _to_device_inputs(prior_n, device=device)

            # ---- TTA: D2/D4 transforms, predict, invert, average -----
            transforms = _tta_transforms(int(tta))
            dtm_acc = None
            logit_acc = None
            for fwd, inv in transforms:
                dm_aug = fwd(dm_t)
                dn_aug = fwd(dn_t)
                dme_aug = fwd(dme_t)
                ma_aug = fwd(ma_t)
                if prior_t is not None:
                    prior_aug = fwd(prior_t)
                    dtm_pred, logit = model.infer(
                        dsm_max=dm_aug, dsm_min=dn_aug,
                        dsm_mean=dme_aug, dsm_mask=ma_aug,
                        init='priostitch', prior_dtm=prior_aug,
                        return_logit=True)
                else:
                    dtm_pred, logit = model.infer(
                        dsm_max=dm_aug, dsm_min=dn_aug,
                        dsm_mean=dme_aug, dsm_mask=ma_aug,
                        init='noisy_dsm', return_logit=True)
                # Un-augment back to canonical orientation, then accumulate
                # on-device to save host roundtrips.
                dtm_pred = inv(dtm_pred)
                logit = inv(logit)
                if dtm_acc is None:
                    dtm_acc = dtm_pred
                    logit_acc = logit
                else:
                    dtm_acc = dtm_acc + dtm_pred
                    logit_acc = logit_acc + logit
            dtm_pred = dtm_acc / float(len(transforms))
            logit = logit_acc / float(len(transforms))

            dtm_pred_np = dtm_pred[0, 0].float().cpu().numpy()
            logit_np = logit[0, 0].float().cpu().numpy()
            # De-normalise back to metres
            dtm_pred_m = (dtm_pred_np + 1.0) / 2.0 * (z_hi - z_lo) + z_lo

            # Logits always blend with the cosine window (smooth seams).
            sum_logit[iy:iy_e, ix:ix_e] += logit_np * logit_window
            sum_w[iy:iy_e, ix:ix_e] += logit_window

            if blend_mode in weighted_modes:
                sum_dtm[iy:iy_e, ix:ix_e] += dtm_pred_m * window
            elif blend_mode == 'mean':
                sum_dtm[iy:iy_e, ix:ix_e] += dtm_pred_m
                cnt_dtm[iy:iy_e, ix:ix_e] += 1
            elif blend_mode == 'min':
                view = dtm_blend[iy:iy_e, ix:ix_e]
                np.minimum(view, dtm_pred_m, out=view)
            elif blend_mode == 'max':
                view = dtm_blend[iy:iy_e, ix:ix_e]
                np.maximum(view, dtm_pred_m, out=view)

            if progress_cb is not None:
                progress_cb(t_done, n_tiles)

    # Final assembly. Logits always use cosine-mean for smooth seams.
    sum_w = np.maximum(sum_w, 1e-9)
    logit_full = (sum_logit / sum_w).astype(np.float32)

    if blend_mode in weighted_modes:
        # The DTM window may be different from logit_window, so we
        # divide by the DTM-side sum of weights. Since the same set of
        # tiles touches each pixel, the per-pixel weight sum is
        # (#tiles_covering · window). We reconstruct it by re-using the
        # tile-origin loop count via the DTM window itself: easier to
        # divide sum_dtm by the running sum of `window` directly, but
        # we already have sum_w for the cosine-windowed logits. Pull
        # the kernel ratio so divisions match per-pixel coverage.
        # Implementation: compute a parallel sum_w_dtm so each weighted
        # mode normalises by its own window.
        sum_w_dtm = np.zeros((H, W), dtype=np.float64)
        for iy in ys:
            for ix in xs:
                iy_e = min(iy + tile_size, H)
                ix_e = min(ix + tile_size, W)
                wh, ww = iy_e - iy, ix_e - ix
                sum_w_dtm[iy:iy_e, ix:ix_e] += window[:wh, :ww]
        sum_w_dtm = np.maximum(sum_w_dtm, 1e-9)
        dtm_pred_full = (sum_dtm / sum_w_dtm).astype(np.float32)
    elif blend_mode == 'mean':
        cnt = np.maximum(cnt_dtm.astype(np.float64), 1.0)
        dtm_pred_full = (sum_dtm / cnt).astype(np.float32)
        # Cells with no tile (shouldn't happen if tiling covers H×W,
        # but defensive): fall back to DSM.
        if (cnt_dtm == 0).any():
            mask = (cnt_dtm == 0)
            dtm_pred_full = np.where(mask, dsm_max.astype(np.float32),
                                       dtm_pred_full)
    else:
        # min / max: cells never touched keep the ±inf sentinel —
        # fill with DSM as a safe finite fallback.
        finite = np.isfinite(dtm_blend)
        if not finite.all():
            dtm_blend = np.where(finite, dtm_blend, dsm_max.astype(np.float64))
        dtm_pred_full = dtm_blend.astype(np.float32)

    # Crop back to the original scene size if we padded a sub-tile scene.
    if (H_orig, W_orig) != (H, W):
        dtm_pred_full = dtm_pred_full[:H_orig, :W_orig]
        logit_full = logit_full[:H_orig, :W_orig]
    return dtm_pred_full, logit_full


@torch.no_grad()
def priostitch_infer(
    model,
    dsm_max: np.ndarray,
    dsm_min: np.ndarray,
    dsm_mean: np.ndarray,
    dsm_mask: np.ndarray,
    *,
    coarse_size: int = 256,
    tile_size: int = 256,
    overlap: int = 128,
    blend_mode: str = 'linear',
    device: str | torch.device = 'cuda',
    use_priostitch: bool = True,
    tta: int = 1,
    progress_cb=None,
) -> dict:
    """End-to-end: coarse pass, then fine tile pass with priostitch init.

    Defaults follow paper §3.3 + §8 Table 7 exactly:
      coarse_size = 512 (network's input dimensions, paper §3.3)
      tile_size   = 512 (matches training)
      overlap     = 256 (paper "tiles overlap by 50 percent" scaled to 512² → stride 256)
      blend_mode  = 'min' (best-performing PrioStitch ablation row)

    Args:
        tta: 1 | 4 | 8 — test-time augmentation passes per fine tile.
             Default 1 (no TTA). 8 = full D4 (paper's stage-1 default
             for the test script). Costs `tta`× fine-pass time.

    Returns dict with:
        dtm_pred:    [H, W] float32 metres
        logit:       [H, W] float32 raw confidence logits
        prob_ground: [H, W] float32 in [0, 1]
        coarse_dtm:  [Hc, Wc] float32 metres (the global prior)
    """
    coarse_dtm_m, z_lo, z_hi = None, None, None
    if use_priostitch:
        coarse_dtm_m, z_lo, z_hi = coarse_pass(
            model, dsm_max, dsm_min, dsm_mean, dsm_mask,
            coarse_size=coarse_size, device=device)
    dtm_full, logit_full = fine_pass(
        model, dsm_max, dsm_min, dsm_mean, dsm_mask,
        coarse_dtm_m=coarse_dtm_m,
        tile_size=tile_size, overlap=overlap, blend_mode=blend_mode,
        device=device,
        use_priostitch=use_priostitch, tta=tta,
        progress_cb=progress_cb)
    prob_ground = 1.0 / (1.0 + np.exp(-logit_full.astype(np.float64))).astype(np.float32)
    return dict(
        dtm_pred=dtm_full,
        logit=logit_full,
        prob_ground=prob_ground.astype(np.float32),
        coarse_dtm=coarse_dtm_m,
    )

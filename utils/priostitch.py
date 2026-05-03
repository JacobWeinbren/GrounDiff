"""PrioStitch (paper §3.3 + Fig.5).

Pipeline at inference time on an arbitrarily-large DSM:

  (a) Downsample DSM to network input size (256×256). Run GrounDiff
      with init=N(s, I) → low-resolution prior DTM.
  (b) Tile the original full-resolution DSM into overlapping 256×256
      patches, stride < 256.
  (c) For each patch: extract the corresponding region of the upsampled
      prior DTM, use that as the diffusion init (q_sample at γ_T).
      Run GrounDiff to produce a refined DTM patch.
  (d) Blend overlapping patches.

Blend modes (paper §12.2 + Tab.7):
  'mean'        — simple average over overlaps
  'min'         — per-cell min (paper's BEST, Tab.7 RMSE 0.514)
  'max'         — per-cell max
  'linear'      — distance-from-edge linear ramp
  'cosine'      — half-cosine ramp (smoothest)
  'exponential' — exp(-d²) like a Gaussian falloff
"""
from __future__ import annotations
import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _make_blend_weights(tile: int, mode: str = 'linear') -> np.ndarray:
    """Per-pixel weight kernel of shape (tile, tile). Higher in the
    centre, falls off toward the edges. Used by weighted-blend modes
    only; min/max/mean are handled separately as per-pixel reductions."""
    coord = (np.arange(tile) + 0.5) / tile        # in (0, 1)
    # Distance from edge along each axis, normalised to [0, 1] (centre = 1)
    dx = np.minimum(coord, 1.0 - coord) * 2.0
    dy = np.minimum(coord, 1.0 - coord) * 2.0
    Y, X = np.meshgrid(dy, dx, indexing='ij')
    d = np.minimum(X, Y)                          # nearest edge

    if mode == 'linear':
        w = d.astype(np.float32)
    elif mode == 'cosine':
        w = (0.5 - 0.5 * np.cos(np.pi * d)).astype(np.float32)
    elif mode == 'exponential':
        # Centre weight 1, edge weight exp(-2) ≈ 0.135
        w = np.exp(-2.0 * (1.0 - d)).astype(np.float32)
    else:
        raise ValueError(f"_make_blend_weights does not handle mode {mode!r} "
                         "(use weighted_blend dispatcher instead)")
    return np.maximum(w, 1e-3)


def weighted_blend(tile_outputs, tile_origins, full_h: int, full_w: int,
                   tile: int = 256, mode: str = 'min'
                   ) -> np.ndarray:
    """Blend a list of tile predictions back into a single full-size
    raster.

    Modes (paper §12.2 + Tab.7):
        'mean':         simple average over overlaps
        'min':          per-cell min — paper's best (Tab.7 RMSE 0.514)
        'max':          per-cell max
        'linear':       distance-from-edge linear weighting
        'cosine':       half-cosine weighting
        'exponential':  exp decay from centre

    Args:
        tile_outputs: list of [tile, tile] np.float32 arrays
        tile_origins: list of (i, j) pixel origins (top-left) per tile
        full_h, full_w: output raster shape
        tile: tile size
        mode: blend mode

    Returns:
        blended full-size raster
    """
    if mode in ('min', 'max', 'mean'):
        # Per-cell reduction over overlapping tiles. No weight kernel.
        out = np.full((full_h, full_w), np.nan, dtype=np.float32)
        if mode == 'mean':
            sum_ = np.zeros((full_h, full_w), dtype=np.float64)
            cnt  = np.zeros((full_h, full_w), dtype=np.int32)
            for arr, (i, j) in zip(tile_outputs, tile_origins):
                i1 = min(i + tile, full_h)
                j1 = min(j + tile, full_w)
                h = i1 - i; w = j1 - j
                sum_[i:i1, j:j1] += arr[:h, :w]
                cnt [i:i1, j:j1] += 1
            mask = cnt > 0
            out[mask] = (sum_[mask] / cnt[mask]).astype(np.float32)
            out[~mask] = 0.0
        else:
            reducer = np.fmin if mode == 'min' else np.fmax
            for arr, (i, j) in zip(tile_outputs, tile_origins):
                i1 = min(i + tile, full_h)
                j1 = min(j + tile, full_w)
                h = i1 - i; w = j1 - j
                cur = out[i:i1, j:j1]
                # np.fmin / np.fmax treat NaN specially: if one side is
                # NaN, the result is the other side. So uninitialised
                # cells get filled, then subsequent overlaps fold in.
                out[i:i1, j:j1] = reducer(cur, arr[:h, :w].astype(np.float32))
            out = np.where(np.isnan(out), 0.0, out)
        return out.astype(np.float32)

    # Weighted-blend modes (linear / cosine / exponential)
    w_kernel = _make_blend_weights(tile, mode=mode)
    out = np.zeros((full_h, full_w), dtype=np.float64)
    den = np.zeros((full_h, full_w), dtype=np.float64)
    for arr, (i, j) in zip(tile_outputs, tile_origins):
        i1 = min(i + tile, full_h)
        j1 = min(j + tile, full_w)
        h = i1 - i; w = j1 - j
        out[i:i1, j:j1] += arr[:h, :w] * w_kernel[:h, :w]
        den[i:i1, j:j1] += w_kernel[:h, :w]
    return (out / np.maximum(den, 1e-9)).astype(np.float32)


def _resize_to(img: np.ndarray, h: int, w: int,
               mode: str = 'bilinear') -> np.ndarray:
    """Resize a 2D float array to (h, w) via torch interpolate.
    Wrapped here to avoid scipy/PIL dependencies."""
    t = torch.from_numpy(img.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(h, w), mode=mode, align_corners=False
                       if mode != 'nearest' else None)
    return t[0, 0].numpy()


def priostitch_inference(model, dsm_full: np.ndarray, *,
                         device: str = 'cuda',
                         tile: int = 256, stride: int = 128,
                         blend_mode: str = 'min',
                         init_prior: bool = True,
                         valid_mask: Optional[np.ndarray] = None,
                         progress: Optional[Callable[[int, int], None]] = None,
                         return_prob: bool = False,
                         dsm_metres: Optional[np.ndarray] = None,
                         scene_stats: Optional[dict] = None,
                         dsm_min_metres: Optional[np.ndarray] = None,
                         capture_tiles: Optional[list] = None,
                         capture_n: int = 4,
                         tta: int = 1,
                         ) -> np.ndarray:
    """Full PrioStitch run with per-tile DSM-only normalisation.

    Args:
        model: GrounDiff instance with .infer()
        dsm_full: [H, W] float32 DSM in [-1, 1] (scene-normalised)
        dsm_metres: [H, W] DSM in metres
        scene_stats: dict with 'vmin'/'vmax' for scene normalisation
        dsm_min_metres: [H, W] optional min-z DSM in metres. If
            provided, fed as 2nd conditioning channel (requires the
            model's UNet to have in_channel=3).
        tile, stride: tiling geometry
        blend_mode: see weighted_blend
        init_prior: if True, use upsampled low-res prior per tile
        valid_mask: optional [H, W] valid mask
        progress: callback(done, total)
        return_prob: also return σ(ℓ) map
        capture_tiles: optional list to APPEND per-tile snapshots into.
        capture_n: how many tiles to snapshot.
        tta: test-time augmentation passes per tile. 1 = no TTA (fastest).
            4 = 4-way D2 (identity, h-flip, v-flip, hv-flip).
            8 = 8-way D4 (above + 90° / 180° / 270° rotations).
            Predictions are averaged after un-augmenting. Kills
            directional/grid bias on tricky scenes (cliffs, structured
            terrain). Costs `tta`× the inference time per tile.

    Returns:
        DTM in METRES, [H, W] float32. (return_prob → tuple.)
    """
    H, W = dsm_full.shape

    # Reconstruct DSM in metres if not provided
    if dsm_metres is None:
        if scene_stats is None:
            raise ValueError("priostitch_inference needs either dsm_metres "
                             "or scene_stats for per-tile re-normalisation")
        s_vmin = float(scene_stats['vmin'])
        s_vmax = float(scene_stats['vmax'])
        s_span = max(s_vmax - s_vmin, 1e-6)
        dsm_metres = (dsm_full + 1.0) * 0.5 * s_span + s_vmin
        if valid_mask is not None:
            dsm_metres = np.where(valid_mask.astype(bool),
                                   dsm_metres, 0.0).astype(np.float32)
    dsm_metres = dsm_metres.astype(np.float32)

    # ---- Step (a): low-res global prior (per-tile DSM-only norm) -------
    # Note: paper Eq.15 jointly normalises by min(s,g) / max(s,g) at
    # training time. We use DSM-only at inference (no GT DTM available
    # on new tiles) AND at training (in dataset.__getitem__) so the
    # train and inference distributions are identical.
    if init_prior:
        dsm_t_full = torch.from_numpy(dsm_metres)[None, None].to(device)
        dsm_lo_m_t = F.interpolate(dsm_t_full, size=(tile, tile),
                                    mode='bilinear', align_corners=False)
        dsm_lo_m = dsm_lo_m_t[0, 0].cpu().numpy()
        if valid_mask is not None:
            vm_full = torch.from_numpy(
                valid_mask.astype(np.float32))[None, None].to(device)
            vm_lo = F.interpolate(vm_full, size=(tile, tile),
                                   mode='bilinear', align_corners=False)
            vm_lo_b = (vm_lo[0, 0].cpu().numpy() > 0.5)
        else:
            vm_lo_b = np.ones((tile, tile), dtype=bool)
        if vm_lo_b.any():
            lo_vmin = float(dsm_lo_m[vm_lo_b].min())
            lo_vmax = float(dsm_lo_m[vm_lo_b].max())
        else:
            lo_vmin, lo_vmax = float(dsm_lo_m.min()), float(dsm_lo_m.max())
        lo_span = max(lo_vmax - lo_vmin, 1e-6)
        dsm_lo_n = np.where(vm_lo_b,
                             2.0 * (dsm_lo_m - lo_vmin) / lo_span - 1.0,
                             0.0).astype(np.float32)
        dsm_lo_t = torch.from_numpy(dsm_lo_n)[None, None].to(device)

        # Optional dsm_min on the low-res prior pass too
        dsm_min_lo_t = None
        if dsm_min_metres is not None:
            dsm_min_full_t = torch.from_numpy(
                dsm_min_metres.astype(np.float32))[None, None].to(device)
            dsm_min_lo_m = F.interpolate(dsm_min_full_t, size=(tile, tile),
                                          mode='bilinear',
                                          align_corners=False
                                          )[0, 0].cpu().numpy()
            dsm_min_lo_n = np.where(
                vm_lo_b,
                np.clip(2.0 * (dsm_min_lo_m - lo_vmin) / lo_span - 1.0,
                         -1.0, 1.0),
                0.0).astype(np.float32)
            dsm_min_lo_t = torch.from_numpy(
                dsm_min_lo_n)[None, None].to(device)

        prior_lo_n = model.infer(dsm_lo_t, init='noisy_dsm',
                                  dsm_min=dsm_min_lo_t)
        prior_lo_m = (prior_lo_n + 1.0) * 0.5 * lo_span + lo_vmin
        prior_full_m = F.interpolate(prior_lo_m, size=(H, W),
                                      mode='bilinear', align_corners=False
                                      )[0, 0].cpu().numpy()
    else:
        prior_full_m = None

    # ---- Step (b)+(c): per-tile diffusion (per-tile normalised) -------
    is_  = list(range(0, max(H - tile, 0) + 1, stride))
    if not is_ or is_[-1] + tile < H: is_.append(max(H - tile, 0))
    js_  = list(range(0, max(W - tile, 0) + 1, stride))
    if not js_ or js_[-1] + tile < W: js_.append(max(W - tile, 0))

    # Decide which tiles to capture for diagnostics. Pick tiles by
    # non-ground content (max-min gap is the proxy when we have
    # dsm_min, otherwise just middle-of-grid). Goal: capture tiles
    # that span boring → interesting so the diagnostic montage
    # is informative.
    n_tiles_total = len(is_) * len(js_)
    capture_indices: set = set()
    if capture_tiles is not None and capture_n > 0:
        # Score each tile by mean canopy-gap on VALID pixels only.
        # Tiles with no valid pixels (entirely off-scene) are excluded
        # from the score pool — they'd produce blank diagnostic panels.
        scores = []
        for ti, i in enumerate(is_):
            for tj, j in enumerate(js_):
                idx = ti * len(js_) + tj
                i1 = min(i + tile, H); j1 = min(j + tile, W)
                if valid_mask is not None:
                    v_block = valid_mask[i:i1, j:j1].astype(bool)
                else:
                    v_block = np.ones((i1 - i, j1 - j), dtype=bool)
                if not v_block.any():
                    # Skip tiles with no valid data — they'd produce
                    # blank panels and the metrics don't depend on them.
                    continue
                if dsm_min_metres is not None:
                    gap = (dsm_metres[i:i1, j:j1]
                           - dsm_min_metres[i:i1, j:j1])
                    sc = float(np.median(gap[v_block]))
                else:
                    region = dsm_metres[i:i1, j:j1][v_block]
                    sc = float(region.max() - region.min())
                scores.append((sc, idx))
        # Sort by score desc, pick top capture_n
        scores.sort(key=lambda x: -x[0])
        capture_indices = {idx for _, idx in scores[:capture_n]}

    outputs_m = []  # in METRES (each tile denormalised by its own stats)
    prob_outputs = []
    origins = []
    total = len(is_) * len(js_)
    done = 0
    for i in is_:
        for j in js_:
            i1 = min(i + tile, H)
            j1 = min(j + tile, W)
            d_m = np.zeros((tile, tile), dtype=np.float32)
            d_m[:i1 - i, :j1 - j] = dsm_metres[i:i1, j:j1]
            v_t = np.zeros((tile, tile), dtype=bool)
            if valid_mask is not None:
                v_t[:i1 - i, :j1 - j] = valid_mask[i:i1, j:j1].astype(bool)
            else:
                v_t[:i1 - i, :j1 - j] = True

            # Per-tile vmin/vmax over valid pixels
            if v_t.any():
                t_vmin = float(d_m[v_t].min())
                t_vmax = float(d_m[v_t].max())
            else:
                t_vmin, t_vmax = 0.0, 1.0
            t_span = max(t_vmax - t_vmin, 1e-6)
            d_n = np.where(v_t,
                            2.0 * (d_m - t_vmin) / t_span - 1.0,
                            0.0).astype(np.float32)
            d_t = torch.from_numpy(d_n)[None, None].to(device)

            # Optional dsm_min channel for this tile
            dsm_min_t_t = None
            if dsm_min_metres is not None:
                dmin_m = np.zeros((tile, tile), dtype=np.float32)
                dmin_m[:i1 - i, :j1 - j] = dsm_min_metres[i:i1, j:j1]
                dmin_n = np.where(
                    v_t,
                    np.clip(2.0 * (dmin_m - t_vmin) / t_span - 1.0,
                             -1.0, 1.0),
                    0.0).astype(np.float32)
                dsm_min_t_t = torch.from_numpy(
                    dmin_n)[None, None].to(device)

            # Construct prior tensor once per tile (TTA reuses it)
            p_m = None
            p_t = None
            if init_prior and prior_full_m is not None:
                p_m = np.zeros((tile, tile), dtype=np.float32)
                p_m[:i1 - i, :j1 - j] = prior_full_m[i:i1, j:j1]
                p_n = np.where(v_t,
                                2.0 * (p_m - t_vmin) / t_span - 1.0,
                                0.0).astype(np.float32)
                p_t = torch.from_numpy(p_n)[None, None].to(device)

            # ---- TTA: run model under D2/D4 transforms and average -----
            # Transforms (numpy spatial flips/rotations) — each entry is
            # (forward_fn, inverse_fn). Forward is applied to the input
            # tensor, inverse to the prediction.
            def _tta_transforms(n: int):
                """Return list of (forward, inverse) tensor transforms.
                n=1 → identity only; n=4 → D2 (4 flips); n=8 → D4."""
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
                    return [(I, I), (hflip, hflip),
                            (vflip, vflip), (hvflip, hvflip)]
                # n=8 → full D4
                return [(I, I), (hflip, hflip), (vflip, vflip),
                        (hvflip, hvflip),
                        (rot1, rot1_inv), (rot2, rot2),
                        (rot3, rot3_inv),
                        # one extra: rot1 ∘ hflip → diagonal reflection
                        (lambda t: torch.rot90(torch.flip(t, [-1]), 1, [-2, -1]),
                         lambda t: torch.flip(torch.rot90(t, -1, [-2, -1]), [-1]))]

            transforms = _tta_transforms(tta)
            preds_n = []
            logits = []
            for fwd, inv in transforms:
                d_t_aug = fwd(d_t)
                dsm_min_aug = fwd(dsm_min_t_t) if dsm_min_t_t is not None else None
                if init_prior and prior_full_m is not None:
                    p_t_aug = fwd(p_t)
                    pred_n_aug, logit_aug = model.infer(
                        d_t_aug, init='priostitch',
                        prior_dtm=p_t_aug,
                        dsm_min=dsm_min_aug,
                        return_logit=True)
                else:
                    pred_n_aug, logit_aug = model.infer(
                        d_t_aug, init='noisy_dsm',
                        dsm_min=dsm_min_aug,
                        return_logit=True)
                # Un-augment back to canonical orientation
                preds_n.append(inv(pred_n_aug))
                logits.append(inv(logit_aug))
            # Average over TTA passes
            pred_n = torch.stack(preds_n, dim=0).mean(dim=0)
            logit  = torch.stack(logits, dim=0).mean(dim=0)

            pred_n_np = pred_n[0, 0].cpu().numpy()
            # Denormalise per-tile back to metres
            pred_m = (pred_n_np + 1.0) * 0.5 * t_span + t_vmin
            outputs_m.append(pred_m.astype(np.float32))
            prob_t = None
            if return_prob:
                prob_t = torch.sigmoid(logit)[0, 0].cpu().numpy()
                prob_outputs.append(prob_t)
            origins.append((i, j))

            # Capture diagnostic snapshot if this tile was chosen
            tile_idx = (is_.index(i) * len(js_)) + js_.index(j)
            if capture_tiles is not None and tile_idx in capture_indices:
                snap = dict(
                    origin=(i, j),
                    tile_size=tile,
                    valid=v_t.copy(),
                    tile_vmin=t_vmin,
                    tile_vmax=t_vmax,
                    dsm_m=d_m.copy(),
                    pred_m=pred_m.astype(np.float32).copy(),
                    prob=(prob_t.copy() if prob_t is not None else None),
                )
                if dsm_min_metres is not None:
                    snap['dsm_min_m'] = dmin_m.copy()
                if init_prior and prior_full_m is not None:
                    snap['prior_m'] = p_m.copy()
                capture_tiles.append(snap)

            done += 1
            if progress is not None:
                progress(done, total)

    # ---- Step (d): weighted blend in metres ---------------------------
    blended = weighted_blend(outputs_m, origins, H, W,
                             tile=tile, mode=blend_mode)
    if valid_mask is not None:
        blended = np.where(valid_mask.astype(bool), blended, 0.0)
    if return_prob:
        prob_blended = weighted_blend(prob_outputs, origins, H, W,
                                       tile=tile, mode='mean')
        if valid_mask is not None:
            prob_blended = np.where(valid_mask.astype(bool), prob_blended, 0.0)
        return blended, prob_blended
    return blended


def save_tile_viz_montage(snapshots: list, *, out_path,
                          dtm_gt_full=None, scene_id: str = "",
                          dsm_min_available: bool = False):
    """Build a per-tile diagnostic montage from priostitch snapshots.

    Each row = one captured tile. Columns vary based on whether
    DSM_min is available:

    Without dsm_min (no min-z channel; baseline):
      DSM | Prior DTM | Pred DTM | σ(ℓ) | Pred − GT (if GT given)

    With dsm_min:
      DSM | DSM_min | Canopy gap | Prior DTM | Pred DTM | Pred − DSM_min | σ(ℓ)

    The "Canopy gap" panel highlights where DSM_min is significantly
    below DSM — the "free ground info" cells. The "Pred − DSM_min"
    panel shows whether the network trusts that signal as ground.

    Args:
        snapshots: list of dicts from `priostitch_inference(capture_tiles=...)`
        out_path: file path for the saved PNG
        dtm_gt_full: optional [H, W] full-scene GT DTM in metres. Used
            to extract per-tile GT and add a final "pred − GT" column.
        scene_id: title prefix
        dsm_min_available: layout flag; if True, render the wide layout.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path

    if not snapshots:
        return None

    if dsm_min_available:
        n_cols = 7 if dtm_gt_full is None else 8
    else:
        n_cols = 4 if dtm_gt_full is None else 5
    n_rows = len(snapshots)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.6 * n_cols, 2.7 * n_rows),
                              squeeze=False)

    for r, snap in enumerate(snapshots):
        i, j = snap['origin']
        tile = snap['tile_size']
        v = snap['valid']
        dsm_m = snap['dsm_m']
        pred_m = snap['pred_m']
        prob = snap.get('prob')
        prior_m = snap.get('prior_m')
        dsm_min_m = snap.get('dsm_min_m')

        def _mask(arr):
            return np.ma.masked_where(~v, arr)

        # Common elev range across DSM, pred, GT (and DSM_min, prior if present)
        rasters = [dsm_m, pred_m]
        if dsm_min_m is not None: rasters.append(dsm_min_m)
        if prior_m is not None: rasters.append(prior_m)
        gt_tile = None
        if dtm_gt_full is not None:
            i1 = min(i + tile, dtm_gt_full.shape[0])
            j1 = min(j + tile, dtm_gt_full.shape[1])
            gt_tile = np.zeros((tile, tile), dtype=np.float32)
            gt_tile[:i1 - i, :j1 - j] = dtm_gt_full[i:i1, j:j1]
            rasters.append(gt_tile)
        if v.any():
            elev_min = float(min(rr[v].min() for rr in rasters))
            elev_max = float(max(rr[v].max() for rr in rasters))
        else:
            elev_min, elev_max = 0.0, 1.0

        col = 0
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])

        # Column 1: DSM
        im = axes[r, col].imshow(_mask(dsm_m), cmap='terrain',
                                  vmin=elev_min, vmax=elev_max)
        axes[r, col].set_title('DSM' if r == 0 else '', fontsize=10)
        axes[r, col].set_ylabel(f"({i},{j})", fontsize=9)
        plt.colorbar(im, ax=axes[r, col], shrink=0.7)
        col += 1

        # DSM_min
        if dsm_min_available:
            if dsm_min_m is not None:
                im = axes[r, col].imshow(_mask(dsm_min_m), cmap='terrain',
                                          vmin=elev_min, vmax=elev_max)
            axes[r, col].set_title('DSM_min' if r == 0 else '', fontsize=10)
            if dsm_min_m is not None:
                plt.colorbar(im, ax=axes[r, col], shrink=0.7)
            col += 1

            # Canopy gap = DSM - DSM_min
            if dsm_min_m is not None:
                gap = dsm_m - dsm_min_m
                gmax = max(1.0, float(np.percentile(gap[v], 99))
                           if v.any() else 1.0)
                im = axes[r, col].imshow(_mask(gap), cmap='YlGn',
                                          vmin=0, vmax=gmax)
                plt.colorbar(im, ax=axes[r, col], shrink=0.7)
            axes[r, col].set_title(
                'DSM − DSM_min' if r == 0 else '', fontsize=10)
            col += 1

        # Prior DTM
        if prior_m is not None:
            im = axes[r, col].imshow(_mask(prior_m), cmap='terrain',
                                      vmin=elev_min, vmax=elev_max)
            plt.colorbar(im, ax=axes[r, col], shrink=0.7)
        axes[r, col].set_title(
            'Prior DTM' if r == 0 else '', fontsize=10)
        col += 1

        # Pred DTM
        im = axes[r, col].imshow(_mask(pred_m), cmap='terrain',
                                  vmin=elev_min, vmax=elev_max)
        axes[r, col].set_title('Pred DTM' if r == 0 else '', fontsize=10)
        plt.colorbar(im, ax=axes[r, col], shrink=0.7)
        col += 1

        # Pred − DSM_min (when available)
        if dsm_min_available and dsm_min_m is not None:
            diff = pred_m - dsm_min_m
            dlim = max(0.5, float(np.percentile(np.abs(diff[v]), 99))
                       if v.any() else 0.5)
            im = axes[r, col].imshow(_mask(diff), cmap='RdBu_r',
                                      vmin=-dlim, vmax=dlim)
            axes[r, col].set_title(
                'Pred − DSM_min' if r == 0 else '', fontsize=10)
            plt.colorbar(im, ax=axes[r, col], shrink=0.7)
            col += 1

        # σ(ℓ)
        if prob is not None:
            im = axes[r, col].imshow(_mask(prob),
                                      cmap='RdYlGn', vmin=0, vmax=1)
            plt.colorbar(im, ax=axes[r, col], shrink=0.7)
        axes[r, col].set_title('σ(ℓ)' if r == 0 else '', fontsize=10)
        col += 1

        # Pred − GT (if GT given)
        if gt_tile is not None:
            err = pred_m - gt_tile
            elim = max(0.5, float(np.percentile(np.abs(err[v]), 99))
                       if v.any() else 0.5)
            im = axes[r, col].imshow(_mask(err), cmap='RdBu_r',
                                      vmin=-elim, vmax=elim)
            axes[r, col].set_title(
                'Pred − GT' if r == 0 else '', fontsize=10)
            plt.colorbar(im, ax=axes[r, col], shrink=0.7)
            col += 1

    fig.suptitle(f"{scene_id} — per-tile diagnostics "
                 f"({len(snapshots)} tiles, ranked by canopy gap)",
                 fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return out_path

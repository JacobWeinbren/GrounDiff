#!/usr/bin/env python3
"""Build separate visualisation artefacts for one scene:

    <scene>_elevation.png   3-panel figure: DSM | DTM_GT | DTM_pred
                            Colourblind-safe hypsometric + hillshade,
                            shared elevation range / colourbar.

    <scene>_analysis.png    2-panel figure: error map | classification
                            Error in BWR (white at zero, ±1m default).
                            Cells without a direct ground return get a
                            light grey background + subtle diagonal
                            hatching to flag interpolated GT.

    <scene>_metrics.csv     all numeric stats — scene-level RMSE / MAE /
                            bias / P95 |e| / fraction within α / cell-
                            level E_T1 / E_T2 / E_tot.

This separation matches the user's request that visualisations be only
about the picture and metrics live in a CSV. Inline stats text is gone
from the images.

Usage:
    python -m stage2_raster.scripts.visualize \\
        --raster path/to/scene/raster.npz \\
        --inference path/to/scene/inference.npz \\
        --out-prefix path/to/output/<scene>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage2_raster.utils.colormap import (
    _interp_cmap, _HYPSO_STOPS, _BWR_STOPS,
    GROUND_RGB, NON_GROUND_RGB, INVALID_RGB,
    hypsometric, diverging_residual, classification_bicolor,
    fill_sentinels, hillshade, shade_blend, hatched_overlay,
)


# ---- Panel helpers ----------------------------------------------------- #

def _setup_panel(ax, rgb, title):
    ax.imshow(rgb, interpolation='nearest', origin='upper')
    ax.set_title(title, fontsize=11, color='#1a1a1a', pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a3a3a')
        spine.set_linewidth(0.6)


def _add_inset_cbar(fig, parent_ax, stops, vlo, vhi, label,
                     tick_count=5):
    pos = parent_ax.get_position()
    cax = fig.add_axes([pos.x0 + pos.width * 0.08,
                         pos.y0 - 0.040,
                         pos.width * 0.84, 0.018])
    gradient = np.linspace(0, 1, 512)[None, :]
    bar = _interp_cmap(stops, gradient)
    cax.imshow(bar, aspect='auto', extent=(vlo, vhi, 0, 1))
    cax.set_yticks([])
    cax.tick_params(axis='x', colors='#1a1a1a', labelsize=9,
                      pad=1.5, length=2.5)
    cax.set_xticks(np.linspace(vlo, vhi, tick_count))
    for sp in cax.spines.values():
        sp.set_edgecolor('#3a3a3a')
        sp.set_linewidth(0.5)
    cax.set_xlabel(label, fontsize=9.5, color='#1a1a1a', labelpad=2)


# ---- Compute stats (used by metrics CSV + analysis panel caption) ----- #

def compute_metrics(
    *,
    dsm_max: np.ndarray,
    gt_dtm: np.ndarray,
    valid_gt: np.ndarray,
    dsm_mask: np.ndarray,
    had_ground: np.ndarray,
    dtm_pred: np.ndarray,
    prob_ground: np.ndarray | None,
    alpha_metres: float = 0.20,
) -> dict:
    """Compute scene-level error + classification stats.

    Returns a dict with all values. Cell-level classification uses M_α
    computed at α=0.20m by default.
    """
    valid_gt = valid_gt.astype(bool)
    dsm_mask = dsm_mask.astype(bool)
    had_g = had_ground.astype(bool) if had_ground is not None else dsm_mask
    err_valid = valid_gt & had_g

    out = dict(
        n_cells=int(dsm_max.size),
        n_cells_valid_gt=int(valid_gt.sum()),
        n_cells_with_return=int(dsm_mask.sum()),
        n_cells_had_ground=int(had_g.sum()),
        n_cells_err_valid=int(err_valid.sum()),
        alpha_metres=float(alpha_metres),
    )

    if err_valid.any():
        e = (dtm_pred - gt_dtm)[err_valid]
        out['rmse_m'] = float(np.sqrt((e ** 2).mean()))
        out['mae_m'] = float(np.abs(e).mean())
        out['bias_m'] = float(e.mean())
        out['p95_abs_err_m'] = float(np.quantile(np.abs(e), 0.95))
        out['frac_within_20cm'] = float((np.abs(e) < 0.20).mean())
        out['frac_within_50cm'] = float((np.abs(e) < 0.50).mean())
        # RMSE split by GROUND-TRUTH class, over the same err_valid cells.
        # Ground = M_α (|s − g_GT| < α); non-ground = the rest. Lets the
        # national map separate "how accurate is the bare earth we keep"
        # (rmse_ground) from "how accurate where we removed structure"
        # (rmse_nonground), which behave very differently on sparse tiles.
        gt_ground_ev = (np.abs(dsm_max - gt_dtm) < alpha_metres)[err_valid]
        eg = e[gt_ground_ev]
        eng = e[~gt_ground_ev]
        out['rmse_ground_m'] = (float(np.sqrt((eg ** 2).mean()))
                                if eg.size else float('nan'))
        out['rmse_nonground_m'] = (float(np.sqrt((eng ** 2).mean()))
                                   if eng.size else float('nan'))
        out['n_err_ground'] = int(eg.size)
        out['n_err_nonground'] = int(eng.size)
    else:
        for k in ('rmse_m', 'mae_m', 'bias_m', 'p95_abs_err_m',
                  'frac_within_20cm', 'frac_within_50cm',
                  'rmse_ground_m', 'rmse_nonground_m'):
            out[k] = float('nan')
        out['n_err_ground'] = 0
        out['n_err_nonground'] = 0

    # Cell-level classification. The paper does NOT state the eval
    # threshold mechanism, so we report BOTH predicted-ground criteria.
    # Both share the same true-ground label M_α (|s − g_GT| < α, α=0.20m):
    #   Method A (residual)  : |s − g_pred| < α   → e_t*_pct
    #       The paper's only defined ground criterion (Eq.14 M_α) applied
    #       to the prediction; matches the per-point LAZ classification.
    #   Method B (σ(ℓ)≥0.5)  : threshold the confidence head → e_t*_pct_logit
    #       σ(ℓ) is the BCE-trained confidence / "Ground Prob." map
    #       (Figs 6, 16). The 0.5 cut is not stated in the paper.
    # E_T1 = retain non-ground = FP/n_ng ; E_T2 = remove ground = FN/n_g
    cls_valid = valid_gt & dsm_mask
    gt_g = valid_gt & (np.abs(dsm_max - gt_dtm) < alpha_metres)
    n_g = int((gt_g & cls_valid).sum())
    n_ng = int(((~gt_g) & cls_valid).sum())
    n_v = int(cls_valid.sum())
    out['n_cells_cls_valid'] = n_v
    out['n_gt_ground'] = n_g
    out['n_gt_non_ground'] = n_ng

    # Method A — residual (primary)
    pred_g_res = np.abs(dsm_max - dtm_pred) < alpha_metres
    fp_a = int((pred_g_res & (~gt_g) & cls_valid).sum())
    fn_a = int(((~pred_g_res) & gt_g & cls_valid).sum())
    out['n_fp'] = fp_a
    out['n_fn'] = fn_a
    out['e_t1_pct']  = 100.0 * fp_a / max(n_ng, 1)
    out['e_t2_pct']  = 100.0 * fn_a / max(n_g, 1)
    out['e_tot_pct'] = 100.0 * (fp_a + fn_a) / max(n_v, 1)

    # Method B — σ(ℓ) ≥ 0.5 (secondary)
    if prob_ground is not None:
        pred_g_log = (prob_ground >= 0.5)
        fp_b = int((pred_g_log & (~gt_g) & cls_valid).sum())
        fn_b = int(((~pred_g_log) & gt_g & cls_valid).sum())
        out['e_t1_pct_logit']  = 100.0 * fp_b / max(n_ng, 1)
        out['e_t2_pct_logit']  = 100.0 * fn_b / max(n_g, 1)
        out['e_tot_pct_logit'] = 100.0 * (fp_b + fn_b) / max(n_v, 1)
    return out


def write_metrics_csv(path: Path, metrics: dict, *, scene: str = ''):
    """Write metrics as a single-row CSV (one file per scene)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['scene'] + list(metrics.keys())
    row = {'scene': scene, **metrics}
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)


def append_metrics_csv(path: Path, metrics: dict, *, scene: str = ''):
    """Append to a multi-scene CSV. Writes header if file is empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['scene'] + list(metrics.keys())
    row = {'scene': scene, **metrics}
    new_file = (not path.exists()) or path.stat().st_size == 0
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow(row)


# ---- Elevation figure (3 panels) -------------------------------------- #

def make_elevation_figure(
    *,
    dsm_max: np.ndarray,
    gt_dtm: np.ndarray,
    valid_gt: np.ndarray,
    dsm_mask: np.ndarray,
    dtm_pred: np.ndarray,
    gsd: float = 0.10,
    title: str = "Elevation",
    out_path: Path | None = None,
    dpi: int = 200,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid_gt = valid_gt.astype(bool)
    dsm_mask = dsm_mask.astype(bool)

    # Shared elevation range across the three panels.
    e_valid = valid_gt & dsm_mask
    if e_valid.any():
        stack = np.concatenate([
            dsm_max[e_valid].ravel(),
            gt_dtm[e_valid].ravel(),
            dtm_pred[e_valid].ravel()])
        vmin = float(np.quantile(stack, 0.02))
        vmax = float(np.quantile(stack, 0.98))
    else:
        vmin, vmax = 0.0, 1.0

    # Fill DSM sentinel cells for display continuity.
    dsm_disp = fill_sentinels(dsm_max, dsm_mask)
    gt_disp = gt_dtm
    pred_disp = dtm_pred

    # Smooth source for hillshading so per-cell noise doesn't speckle.
    try:
        from scipy.ndimage import gaussian_filter
        dsm_sh = gaussian_filter(dsm_disp.astype(np.float32), sigma=1.0)
        gt_sh = gaussian_filter(gt_disp.astype(np.float32), sigma=1.0)
        pred_sh = gaussian_filter(pred_disp.astype(np.float32), sigma=1.0)
    except ImportError:
        dsm_sh, gt_sh, pred_sh = dsm_disp, gt_disp, pred_disp

    shade_dsm = hillshade(dsm_sh, gsd=gsd, z_factor=1.5)
    shade_gt = hillshade(gt_sh, gsd=gsd, z_factor=1.5)
    shade_pred = hillshade(pred_sh, gsd=gsd, z_factor=1.5)

    # Display masking — all three panels share the SAME extent so the
    # coastline/hull contour is consistent across DSM | GT | Pred.
    #
    # The shared extent is the TIN convex hull `valid_gt` (the cells where
    # the GT DTM is defined by TIN-interpolation of ground returns). This
    # is the meaningful data region: outside the hull there is no GT and
    # nothing to compare against.
    #
    #   DSM panel : TIN-extent (valid_gt). The DSM itself is rasterised
    #               max-z; we mask it to the same hull as GT so the two
    #               input/target panels are directly comparable rather
    #               than the DSM showing its own (looser) per-cell return
    #               coverage.
    #   GT panel  : TIN DTM masked to valid_gt (its native extent).
    #   Pred panel: the FULL model output (the network predicts a DTM
    #               value at every cell, including outside the hull),
    #               then masked to valid_gt for display so it shares the
    #               same contour. The full output is preserved in
    #               `dtm_pred`; masking is display-only.
    tin_extent = valid_gt.astype(bool)

    rgb_dsm = shade_blend(
        hypsometric(dsm_disp, vmin=vmin, vmax=vmax, valid=tin_extent),
        shade_dsm, strength=0.45)
    rgb_gt = shade_blend(
        hypsometric(gt_disp, vmin=vmin, vmax=vmax, valid=tin_extent),
        shade_gt, strength=0.45)
    rgb_pred = shade_blend(
        hypsometric(pred_disp, vmin=vmin, vmax=vmax, valid=tin_extent),
        shade_pred, strength=0.45)

    H_px, W_px = dsm_max.shape
    aspect = W_px / H_px
    fig_w = 18.0
    panel_h = (fig_w / 3) / aspect
    fig_h = panel_h + 1.2
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor('#fafafa')
    gs = fig.add_gridspec(
        1, 3, hspace=0.05, wspace=0.06,
        left=0.02, right=0.985, top=0.90, bottom=0.20)

    ax_dsm = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_pred = fig.add_subplot(gs[0, 2])
    _setup_panel(ax_dsm, rgb_dsm, "DSM (max z)")
    _setup_panel(ax_gt, rgb_gt, "DTM — ground truth")
    _setup_panel(ax_pred, rgb_pred, "DTM — predicted")

    _add_inset_cbar(fig, ax_dsm, _HYPSO_STOPS, vmin, vmax, "Elevation (m)")
    _add_inset_cbar(fig, ax_gt, _HYPSO_STOPS, vmin, vmax, "Elevation (m)")
    _add_inset_cbar(fig, ax_pred, _HYPSO_STOPS, vmin, vmax, "Elevation (m)")

    fig.suptitle(title, fontsize=13, color='#1a1a1a',
                  fontweight='bold', y=0.98)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(),
                     bbox_inches='tight', pad_inches=0.15)
        print(f"  elevation -> {out_path}")
        plt.close(fig)        # release; pyplot retains figures until closed
        return None
    return fig


# ---- Analysis figure (error + classification) ------------------------- #

def make_analysis_figure(
    *,
    dsm_max: np.ndarray,
    gt_dtm: np.ndarray,
    valid_gt: np.ndarray,
    dsm_mask: np.ndarray,
    dtm_pred: np.ndarray,
    had_ground: np.ndarray | None = None,
    prob_ground: np.ndarray | None = None,
    logit: np.ndarray | None = None,
    gsd: float = 0.10,
    title: str = "Analysis",
    err_vrange: float = 1.0,
    out_path: Path | None = None,
    dpi: int = 200,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    valid_gt = valid_gt.astype(bool)
    dsm_mask = dsm_mask.astype(bool)
    if had_ground is None:
        had_g = dsm_mask
    else:
        had_g = had_ground.astype(bool)

    # ---- Error panel ---------------------------------------------------
    err = dtm_pred - gt_dtm
    err_show = valid_gt & had_g
    # Where dsm_mask but not had_ground (i.e. cells with returns but no
    # ground class) -> GT is interpolated, hatch them.
    interp_mask = valid_gt & (~had_g) & dsm_mask
    # Where dsm_mask=False -> no data at all, use INVALID_RGB.
    no_data_mask = ~dsm_mask
    rgb_err = diverging_residual(err, vrange=err_vrange, valid=err_show)
    rgb_err = hatched_overlay(rgb_err, interp_mask, line_spacing=7)
    # Apply light grey to "no data" cells (overriding any colour they have).
    rgb_err[no_data_mask] = np.array(INVALID_RGB, dtype=np.float32)

    # ---- Classification panel -----------------------------------------
    if prob_ground is None and logit is not None:
        prob_ground = 1.0 / (1.0 + np.exp(-logit.astype(np.float64)))
        prob_ground = prob_ground.astype(np.float32)
    if prob_ground is None:
        prob_ground = np.full_like(dsm_max, 0.5, dtype=np.float32)
    pred_g = (prob_ground >= 0.5)
    # Classification shown over the TIN hull (valid_gt) — same extent as
    # the elevation panels — so the displayed region is consistent across
    # all figures. (Cells outside the hull have no GT to classify against.)
    rgb_class = classification_bicolor(pred_g, valid=valid_gt)

    # Light hillshading on the classification for terrain context.
    try:
        from scipy.ndimage import gaussian_filter
        dsm_sh_src = gaussian_filter(
            fill_sentinels(dsm_max, dsm_mask).astype(np.float32), sigma=1.0)
    except ImportError:
        dsm_sh_src = fill_sentinels(dsm_max, dsm_mask)
    shade = hillshade(dsm_sh_src, gsd=gsd, z_factor=1.5)
    rgb_class = shade_blend(rgb_class, shade, strength=0.18)

    # ---- Figure layout (2 panels) -------------------------------------
    H_px, W_px = dsm_max.shape
    aspect = W_px / H_px
    fig_w = 13.0
    panel_h = (fig_w / 2) / aspect
    # Extra bottom space: colourbar + colourbar label + the hatching note
    # need to live below the panel without colliding.
    fig_h = panel_h + 2.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor('#fafafa')
    gs = fig.add_gridspec(
        1, 2, hspace=0.05, wspace=0.06,
        left=0.025, right=0.985, top=0.92, bottom=0.26)

    ax_err = fig.add_subplot(gs[0, 0])
    ax_cls = fig.add_subplot(gs[0, 1])
    _setup_panel(ax_err, rgb_err,
                  f"Error  DTM_pred − DTM_GT   (±{err_vrange:.1f} m)")
    _setup_panel(ax_cls, rgb_class,
                  "Classification (cell-level)")

    _add_inset_cbar(fig, ax_err, _BWR_STOPS, -err_vrange, err_vrange,
                     "Error (m)")

    # Classification legend (chip strip under the panel). Pushed down so
    # it lines up with the error colourbar height visually.
    pos = ax_cls.get_position()
    cax = fig.add_axes([pos.x0 + pos.width * 0.05,
                         pos.y0 - 0.060,
                         pos.width * 0.90, 0.030])
    cax.set_xlim(0, 1); cax.set_ylim(0, 1); cax.axis('off')
    chip_y0 = 0.15; chip_h = 0.70
    cax.add_patch(Rectangle((0.02, chip_y0), 0.022, chip_h,
                              facecolor=GROUND_RGB, edgecolor='#3a3a3a',
                              linewidth=0.5))
    cax.text(0.058, chip_y0 + chip_h/2, "ground  (σ(ℓ) ≥ 0.5)",
              fontsize=10, color='#1a1a1a', va='center')
    cax.add_patch(Rectangle((0.38, chip_y0), 0.022, chip_h,
                              facecolor=NON_GROUND_RGB, edgecolor='#3a3a3a',
                              linewidth=0.5))
    cax.text(0.418, chip_y0 + chip_h/2, "non-ground",
              fontsize=10, color='#1a1a1a', va='center')
    cax.add_patch(Rectangle((0.65, chip_y0), 0.022, chip_h,
                              facecolor=INVALID_RGB, edgecolor='#3a3a3a',
                              linewidth=0.5))
    cax.text(0.688, chip_y0 + chip_h/2, "no data / invalid",
              fontsize=10, color='#1a1a1a', va='center')

    # Note about hatching — well below the error colourbar to avoid the
    # collision the previous layout had. We attach it to the figure
    # (not to ax_err) and centre it under the error panel column.
    err_pos = ax_err.get_position()
    note_ax = fig.add_axes(
        [err_pos.x0,
          err_pos.y0 - 0.140,
          err_pos.width, 0.040])
    note_ax.axis('off')
    note_ax.text(0.5, 0.5,
                  "hatched = GT was interpolated (no ground return in cell)        "
                  "light grey = no ALS data in cell",
                  fontsize=8.5, color='#3a3a3a',
                  ha='center', va='center',
                  transform=note_ax.transAxes)

    fig.suptitle(title, fontsize=13, color='#1a1a1a',
                  fontweight='bold', y=0.98)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(),
                     bbox_inches='tight', pad_inches=0.15)
        print(f"  analysis  -> {out_path}")
        plt.close(fig)        # release; pyplot retains figures until closed
        return None
    return fig


# ---- Combined entry point -------------------------------------------- #

def make_all(
    *,
    dsm_max: np.ndarray,
    gt_dtm: np.ndarray,
    valid_gt: np.ndarray,
    dsm_mask: np.ndarray,
    dtm_pred: np.ndarray,
    had_ground: np.ndarray | None = None,
    prob_ground: np.ndarray | None = None,
    logit: np.ndarray | None = None,
    gsd: float = 0.10,
    alpha_metres: float = 0.20,
    title_prefix: str = "stage2_raster",
    err_vrange: float = 1.0,
    out_prefix: Path | str,
    dpi: int = 200,
    scene_name: str = '',
    rollup_csv: Path | None = None,
) -> dict:
    """Render both figures and write the metrics CSV next to them.

    Returns the metrics dict.
    """
    out_prefix = Path(out_prefix)
    # Per-scene metrics + (optional) multi-scene rollup.
    metrics = compute_metrics(
        dsm_max=dsm_max, gt_dtm=gt_dtm, valid_gt=valid_gt,
        dsm_mask=dsm_mask, had_ground=had_ground,
        dtm_pred=dtm_pred,
        prob_ground=(prob_ground if prob_ground is not None
                     else (1.0 / (1.0 + np.exp(-logit.astype(np.float64))))
                            .astype(np.float32) if logit is not None else None),
        alpha_metres=alpha_metres)
    write_metrics_csv(out_prefix.with_name(out_prefix.name + '_metrics.csv'),
                      metrics, scene=scene_name)
    print(f"  metrics   -> {out_prefix.name}_metrics.csv  "
          f"(RMSE={metrics.get('rmse_m', float('nan')):.3f} m,  "
          f"E_tot={metrics.get('e_tot_pct', float('nan')):.2f} %)")
    if rollup_csv is not None:
        append_metrics_csv(rollup_csv, metrics, scene=scene_name)

    make_elevation_figure(
        dsm_max=dsm_max, gt_dtm=gt_dtm, valid_gt=valid_gt,
        dsm_mask=dsm_mask, dtm_pred=dtm_pred, gsd=gsd,
        title=f"{title_prefix}  •  Elevation  •  {scene_name}",
        out_path=out_prefix.with_name(out_prefix.name + '_elevation.png'),
        dpi=dpi)

    make_analysis_figure(
        dsm_max=dsm_max, gt_dtm=gt_dtm, valid_gt=valid_gt,
        dsm_mask=dsm_mask, dtm_pred=dtm_pred,
        had_ground=had_ground, prob_ground=prob_ground, logit=logit,
        gsd=gsd,
        title=f"{title_prefix}  •  Analysis  •  {scene_name}",
        err_vrange=err_vrange,
        out_path=out_prefix.with_name(out_prefix.name + '_analysis.png'),
        dpi=dpi)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raster', required=True)
    ap.add_argument('--inference', required=True,
                    help="Inference .npz (dtm_pred, logit, prob_ground).")
    ap.add_argument('--out-prefix', required=True,
                    help="Output prefix path, e.g. '/path/out/scene123'. "
                         "Produces <prefix>_elevation.png, "
                         "<prefix>_analysis.png, <prefix>_metrics.csv.")
    ap.add_argument('--title', default="stage2_raster")
    ap.add_argument('--err-vrange', type=float, default=1.0)
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--scene-name', default='')
    ap.add_argument('--rollup-csv', default=None,
                    help="If set, append this scene's metrics to this CSV "
                         "(useful for batch viz over many scenes).")
    args = ap.parse_args()

    with np.load(args.raster) as z:
        dsm_max = z['dsm_max']
        dsm_mask = z['dsm_mask']
        gt_dtm = z['gt_dtm']
        valid = z['valid']
        had_ground = z.get('had_ground_return')
        gsd = float(z['gsd'])
        alpha = float(z['alpha_metres'])
    with np.load(args.inference) as z:
        dtm_pred = z['dtm_pred']
        logit = z['logit'] if 'logit' in z.files else None
        prob_ground = (z['prob_ground'] if 'prob_ground' in z.files
                        else None)

    scene = args.scene_name or Path(args.raster).parent.name
    rollup = Path(args.rollup_csv) if args.rollup_csv else None
    make_all(
        dsm_max=dsm_max, gt_dtm=gt_dtm,
        valid_gt=valid, dsm_mask=dsm_mask,
        dtm_pred=dtm_pred,
        had_ground=had_ground,
        logit=logit, prob_ground=prob_ground,
        gsd=gsd, alpha_metres=alpha,
        title_prefix=args.title, err_vrange=args.err_vrange,
        out_prefix=args.out_prefix, dpi=args.dpi,
        scene_name=scene, rollup_csv=rollup)


if __name__ == '__main__':
    main()

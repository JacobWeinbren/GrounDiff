"""Test / evaluation script for GrounDiff with publication-grade outputs.

Two output classes:
  1. metrics.csv — per-scene + global metrics
  2. viz/<scene>.png — high-resolution paper-style triptych:
     [DSM | GT DTM | Pred DTM] side-by-side, hillshaded, large

Usage examples:

  # Full England-wide sweep (stratified by OS grid 100km square),
  # paper-style triptych viz on the largest scenes:
  python -u -m scripts.test \\
      --config configs/defra.json \\
      --tile_dir /data/england_tiles \\
      --resume <ckpt>.pt \\
      --out_dir runs/test_full_england \\
      --split all \\
      --n_scenes 200 \\
      --viz_top_n 16

  # Quick test-split eval, no viz:
  python -u -m scripts.test \\
      --config configs/defra.json \\
      --tile_dir /data/england_tiles \\
      --resume <ckpt>.pt \\
      --out_dir runs/test_eval \\
      --no_viz

Speed: PrioStitch on ~4000×4000 scene ≈ 90-150s on Blackwell.
200 scenes ≈ 5-8 hours. 953 scenes ≈ 24-40 hours.
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.normalize import denormalise              # noqa: E402
from models import GrounDiff, MetricAggregator      # noqa: E402
from models.metrics import (                          # noqa: E402
    classify_points_against_dtm, per_point_classification)
from utils import priostitch_inference              # noqa: E402

GROUND_CLASSES = (2, 9)   # LAS cls 2 (ground) ∪ 9 (water); see DECISIONS.md


# ============================================================================
# Helpers — LAZ reading, scene loading, logger
# ============================================================================

def _read_laz_points(path: Path):
    """Light LAZ read returning (x, y, z, classification). Returns None
    on corrupted files."""
    try:
        import laspy
    except ImportError:
        return None
    try:
        with laspy.open(str(path)) as r:
            if r.header.point_count < 100:
                return None
            las = r.read()
        return (np.asarray(las.x, dtype=np.float64),
                np.asarray(las.y, dtype=np.float64),
                np.asarray(las.z, dtype=np.float64),
                np.asarray(las.classification, dtype=np.uint8))
    except Exception:
        return None


def _setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('test')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                             datefmt='%H:%M:%S')
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / 'test.log'); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def _load_scene(scene_marker: Path):
    with np.load(str(scene_marker), allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    if 'stats' in d and isinstance(d['stats'], np.ndarray) \
            and d['stats'].dtype == object:
        d['stats'] = d['stats'].item()
    return d


# ============================================================================
# Publication-grade visualisation
# ============================================================================

def _terrain_colormap():
    """Custom paper-style elevation ramp: dark green → light green →
    yellow → orange → reddish-brown → near-white. Matches the visual
    feel of Figs. 9-11 in the GrounDiff paper.

    Cached on first use.
    """
    from matplotlib.colors import LinearSegmentedColormap
    if not hasattr(_terrain_colormap, '_cmap'):
        # Hand-tuned stops to approximate paper figures
        stops = [
            (0.00, '#2a4d2e'),  # very low — dark green / forest shade
            (0.12, '#3a7a3f'),  # low — mid green
            (0.28, '#7fb05a'),  # rolling — light green
            (0.45, '#dac24d'),  # mid — yellow / wheat
            (0.62, '#d68a36'),  # higher — orange / amber
            (0.80, '#9c5a2c'),  # high — reddish brown
            (1.00, '#e8d8c0'),  # peaks — pale beige
        ]
        positions = [s[0] for s in stops]
        colors = [s[1] for s in stops]
        _terrain_colormap._cmap = LinearSegmentedColormap.from_list(
            'paper_terrain', list(zip(positions, colors)), N=512)
    return _terrain_colormap._cmap


def _inpaint_invalid(arr, valid, *, max_radius=8):
    """Fill small invalid regions in `arr` so the visualisation looks
    continuous (paper-style). Uses iterative dilation with mean of
    valid neighbours. Returns (filled, filled_mask) where filled_mask
    is True for pixels that are now valid (originally valid OR filled).

    Only fills gaps narrower than ~max_radius pixels in either dim;
    larger holes stay invalid (they're meant to look like ocean/edge).
    """
    from scipy.ndimage import distance_transform_edt
    filled = arr.copy().astype(np.float32)
    v = valid.astype(bool).copy()
    if v.all():
        return filled, v
    # Distance to nearest valid pixel (in pixels).
    # `indices=True` returns the index of the nearest valid for each cell.
    inv = ~v
    dist, (idx_i, idx_j) = distance_transform_edt(
        inv, return_distances=True, return_indices=True)
    # For pixels with distance ≤ max_radius, fill from nearest valid neighbour.
    fill_mask = inv & (dist <= max_radius)
    if fill_mask.any():
        filled[fill_mask] = arr[idx_i[fill_mask], idx_j[fill_mask]]
        v = v | fill_mask
    return filled, v


def _shade_terrain(elev, cmap, vmin, vmax, *,
                    azimuth=315.0, altitude=35.0,
                    vertical_exaggeration=4.0,
                    blend_mode='overlay'):
    """Apply hillshade to a coloured elevation map, paper-style.

    Defaults tuned to match GrounDiff paper Figs 9-11:
    - azimuth 315° (NW light source — geo-viz convention)
    - altitude 35° (relatively low sun for stronger shadows)
    - vertical_exaggeration 4× (terrain looks dramatic)
    - blend_mode 'overlay' (more saturated than 'soft')

    Returns an [H, W, 4] RGBA float array in [0, 1]. Caller is
    responsible for masking invalid pixels post-hoc.
    """
    from matplotlib.colors import LightSource, Normalize
    norm = Normalize(vmin=vmin, vmax=vmax)
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    rgb = ls.shade(
        elev, cmap=cmap, norm=norm, blend_mode=blend_mode,
        vert_exag=vertical_exaggeration,
        dx=1.0, dy=1.0,
    )
    if rgb.shape[-1] == 3:
        rgba = np.concatenate(
            [rgb, np.ones(rgb.shape[:-1] + (1,), dtype=rgb.dtype)],
            axis=-1)
    else:
        rgba = rgb
    return rgba.astype(np.float32)


def _ensure_font(family_preferred=('Inter', 'IBM Plex Sans',
                                     'Source Sans Pro', 'Helvetica Neue',
                                     'Helvetica', 'Arial', 'DejaVu Sans')):
    """Return a font family name to use, attempting to download Inter
    if no good system font is installed.

    Datawrapper, NYT, FT, Bloomberg etc. all use Inter or IBM Plex.
    DejaVu Sans (matplotlib's default) looks dated. Try to obtain a
    proper sans-serif, falling back gracefully.
    """
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for fam in family_preferred:
        if fam in available:
            return fam

    # Try to download Inter Variable from rsms.me (small, single file)
    import os, tempfile, urllib.request
    cache_dir = Path(os.path.expanduser('~/.cache/groundiff_fonts'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    inter_path = cache_dir / 'Inter-Regular.otf'
    inter_bold_path = cache_dir / 'Inter-SemiBold.otf'
    if not inter_path.exists():
        try:
            urls = [
                ('https://github.com/rsms/inter/raw/v3.19/docs/font-files/'
                 'Inter-Regular.otf', inter_path),
                ('https://github.com/rsms/inter/raw/v3.19/docs/font-files/'
                 'Inter-SemiBold.otf', inter_bold_path),
            ]
            for url, dest in urls:
                if not dest.exists():
                    urllib.request.urlretrieve(url, dest)
        except Exception:
            return 'DejaVu Sans'
    if inter_path.exists():
        try:
            font_manager.fontManager.addfont(str(inter_path))
            if inter_bold_path.exists():
                font_manager.fontManager.addfont(str(inter_bold_path))
            # Refresh the cache so matplotlib finds the new family
            return 'Inter'
        except Exception:
            return 'DejaVu Sans'
    return 'DejaVu Sans'


def _osgb_to_latlon_str(bbox) -> str:
    """Convert an OS National Grid bbox to a 'lat°N, lon°W' string at
    the bbox centre. Returns empty string on failure (e.g. pyproj not
    installed, bbox is None, or bbox is malformed).

    DEFRA LIDAR scenes are stored with bboxes in EPSG:27700 (OSGB36 /
    British National Grid). We convert the centre to WGS84 (EPSG:4326)
    for human-readable display.
    """
    if bbox is None:
        return ''
    try:
        b = list(bbox)
        if len(b) != 4:
            return ''
        cx = (float(b[0]) + float(b[2])) / 2
        cy = (float(b[1]) + float(b[3])) / 2
    except Exception:
        return ''
    try:
        from pyproj import Transformer
        if not hasattr(_osgb_to_latlon_str, '_t'):
            _osgb_to_latlon_str._t = Transformer.from_crs(
                'EPSG:27700', 'EPSG:4326', always_xy=True)
        lon, lat = _osgb_to_latlon_str._t.transform(cx, cy)
    except ImportError:
        # pyproj not available — silently skip (subtitle still shows
        # extent and pixel count).
        return ''
    except Exception:
        return ''
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return ''
    ns = 'N' if lat >= 0 else 'S'
    ew = 'E' if lon >= 0 else 'W'
    return f"{abs(lat):.4f}°{ns},  {abs(lon):.4f}°{ew}"


def _render_triptych(scene_id: str, dsm_m, dtm_gt_m, pred_m, valid,
                     out_path: Path,
                     *,
                     dpi: int = 200,
                     panel_inches: float = 8.0,
                     stats: dict | None = None,
                     scene_meta: dict | None = None,
                     hillshade: bool = True,
                     inpaint: bool = True,
                     vertical_exaggeration: float = 4.0):
    """Datawrapper / paper-style 4-panel figure:
        DSM | GT DTM | Pred DTM | Error

    The first three panels share an elevation colour ramp with hillshade.
    The fourth panel is a diverging red-blue map of (pred − truth)
    clipped to ±1.5 m, showing where the model over-predicts (red) or
    under-predicts (blue) ground elevation.

    Layout (top → bottom):
        ┌─ Title (scene id, large semibold)
        │  ─ Subtitle (extent + GSD, smaller, grey)
        │
        │  ┌────┐ ┌────┐ ┌────┐ ┌────┐
        │  │DSM │ │GT  │ │Pred│ │Err │       panel labels below
        │  └────┘ └────┘ └────┘ └────┘
        │   DSM   GT DTM  Pred DTM  Error
        │
        │   ─── Elevation cbar ───   ── Error cbar ──
        │   Elevation (m a.s.l.)     Error (m, pred − truth)
        │
        └─ Caption (RMSE, MAE, E_T1, E_T2 — small, monospaced numbers)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # Pick a clean sans-serif (downloads Inter if needed)
    chosen = _ensure_font()
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   [chosen, 'DejaVu Sans'],
        'figure.facecolor':  'white',
        'axes.facecolor':    'white',
        'axes.edgecolor':    '#dddddd',
        'axes.linewidth':    0.6,
        'xtick.color':       '#666666',
        'ytick.color':       '#666666',
        'text.color':        '#222222',
    })

    cmap = _terrain_colormap()
    v0 = valid.astype(bool)

    # Optional inpaint to look paper-clean
    if inpaint:
        dsm_f, v_dsm = _inpaint_invalid(dsm_m, v0, max_radius=8)
        gt_f, v_gt = _inpaint_invalid(dtm_gt_m, v0, max_radius=8)
        pr_f, v_pr = _inpaint_invalid(pred_m, v0, max_radius=8)
        v_disp = v_dsm & v_gt & v_pr
    else:
        dsm_f, gt_f, pr_f = dsm_m, dtm_gt_m, pred_m
        v_disp = v0

    # Robust elevation range
    if v0.any():
        all_m = np.concatenate([dsm_f[v0], gt_f[v0], pr_f[v0]])
        lo, hi = np.percentile(all_m, [0.5, 99.5])
        # Pick a tick step (1, 2, 5, 10, 20, 50 m) so colour-bar labels
        # are clean. Aim for ~6-8 ticks.
        rng = hi - lo
        for step in (1, 2, 5, 10, 20, 50, 100, 200, 500):
            if rng / step <= 8:
                break
        elev_lo = float(np.floor(lo / step) * step)
        elev_hi = float(np.ceil(hi / step) * step)
        if elev_hi <= elev_lo:
            elev_hi = elev_lo + step
    else:
        elev_lo, elev_hi, step = 0.0, 1.0, 1

    norm = Normalize(vmin=elev_lo, vmax=elev_hi)

    def _render_panel(arr2d):
        if hillshade:
            rgba = _shade_terrain(
                arr2d, cmap, elev_lo, elev_hi,
                vertical_exaggeration=vertical_exaggeration,
                altitude=35.0,
                blend_mode='overlay')
        else:
            rgba = cmap(norm(arr2d))
        rgba = rgba.copy()
        rgba[~v_disp] = [1.0, 1.0, 1.0, 1.0]
        return rgba

    # ---- Build the four panels --------------------------------------
    # Panels 1-3 use the elevation cmap with hillshade.
    # Panel 4 (error) uses a diverging cmap, no hillshade, ±1.5m clipped.
    err_max = 1.5
    from matplotlib.colors import Normalize as _Norm

    err_arr = (pr_f - gt_f).astype(np.float32)
    err_arr_clipped = np.clip(err_arr, -err_max, err_max)
    err_cmap = plt.get_cmap('RdBu_r')   # blue (under) → white (zero) → red (over)
    err_norm = _Norm(vmin=-err_max, vmax=err_max)

    err_rgba = err_cmap(err_norm(err_arr_clipped))
    err_rgba = np.ascontiguousarray(err_rgba)
    err_rgba[~v_disp] = [1.0, 1.0, 1.0, 1.0]

    panels = [
        ('Surface model (DSM)',  _render_panel(dsm_f)),
        ('Ground truth (DTM)',   _render_panel(gt_f)),
        ('GrounDiff prediction', _render_panel(pr_f)),
        ('Prediction error',     err_rgba),
    ]
    n_panels = len(panels)

    # ---- LAYOUT ---------------------------------------------------------
    # Allocate vertical space carefully so nothing overlaps.
    title_h, subtitle_h, spacer_h = 0.55, 0.30, 0.20
    panel_label_h, cbar_h, cbar_lbl_h = 0.40, 0.18, 0.55
    caption_h = 0.40 if stats is not None else 0.0
    bottom_margin = 0.25
    side_margin = 0.30
    panel_gap = 0.12  # inches between panels

    fig_w = (panel_inches * n_panels
              + panel_gap * (n_panels - 1)
              + side_margin * 2)
    fig_h = (title_h + subtitle_h + spacer_h
              + panel_inches + panel_label_h
              + cbar_h + cbar_lbl_h
              + caption_h + bottom_margin)

    fig = plt.figure(figsize=(fig_w, fig_h))

    def _y(h_from_top):
        return 1.0 - h_from_top / fig_h

    y_title    = _y(title_h * 0.55)
    y_subtitle = _y(title_h + subtitle_h * 0.55)
    y_panels_top = _y(title_h + subtitle_h + spacer_h)
    y_panels_bot = _y(title_h + subtitle_h + spacer_h + panel_inches)
    y_panel_lbl  = _y(title_h + subtitle_h + spacer_h
                       + panel_inches + panel_label_h * 0.55)
    y_cbar_top   = _y(title_h + subtitle_h + spacer_h
                       + panel_inches + panel_label_h)
    y_cbar_bot   = _y(title_h + subtitle_h + spacer_h
                       + panel_inches + panel_label_h + cbar_h)
    y_cbar_lbl   = _y(title_h + subtitle_h + spacer_h
                       + panel_inches + panel_label_h
                       + cbar_h + cbar_lbl_h * 0.75)
    y_caption    = _y(title_h + subtitle_h + spacer_h
                       + panel_inches + panel_label_h
                       + cbar_h + cbar_lbl_h + caption_h * 0.55)

    panel_w_frac = panel_inches / fig_w
    margin_frac = side_margin / fig_w
    gap_frac = panel_gap / fig_w

    # Panel x-positions (left edges)
    panel_x_lefts = [margin_frac + i * (panel_w_frac + gap_frac)
                     for i in range(n_panels)]

    # Build the 4 panel axes
    panel_axes = []
    for x0 in panel_x_lefts:
        ax = fig.add_axes(
            [x0, y_panels_bot, panel_w_frac,
             y_panels_top - y_panels_bot])
        panel_axes.append(ax)

    for ax, (title, rgba) in zip(panel_axes, panels):
        ax.imshow(rgba, interpolation='nearest', aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.7)

    # Panel labels below each panel
    for i, (title, _) in enumerate(panels):
        x_centre = panel_x_lefts[i] + panel_w_frac / 2
        fig.text(x_centre, y_panel_lbl, title,
                  ha='center', va='center',
                  fontsize=13, weight='semibold', color='#1a1a1a')

    # ---- TWO COLOURBARS -------------------------------------------------
    # 1. Elevation colorbar — centred under the middle of panels 1-3
    elev_cbar_centre = (panel_x_lefts[0]
                         + panel_x_lefts[2] + panel_w_frac) / 2
    elev_cbar_w = panel_w_frac * 1.5
    elev_cbar_x = elev_cbar_centre - elev_cbar_w / 2
    elev_cbar_ax = fig.add_axes(
        [elev_cbar_x, y_cbar_bot, elev_cbar_w,
         y_cbar_top - y_cbar_bot])
    sm_elev = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_elev.set_array([])
    cb_elev = fig.colorbar(sm_elev, cax=elev_cbar_ax,
                            orientation='horizontal')
    cb_elev.outline.set_linewidth(0)
    ticks = np.arange(elev_lo, elev_hi + step * 0.5, step)
    if len(ticks) > 9:
        ticks = ticks[::2]
    cb_elev.set_ticks(ticks)
    cb_elev.ax.tick_params(labelsize=9.5, colors='#444444',
                            length=0, pad=4)
    for spine in cb_elev.ax.spines.values():
        spine.set_visible(False)

    fig.text(elev_cbar_centre, y_cbar_lbl,
              'Elevation (metres above sea level)',
              ha='center', va='center', fontsize=10.5,
              color='#666666', style='italic')

    # 2. Error colorbar — centred under panel 4
    err_cbar_centre = panel_x_lefts[3] + panel_w_frac / 2
    err_cbar_w = panel_w_frac * 0.85
    err_cbar_x = err_cbar_centre - err_cbar_w / 2
    err_cbar_ax = fig.add_axes(
        [err_cbar_x, y_cbar_bot, err_cbar_w,
         y_cbar_top - y_cbar_bot])
    sm_err = plt.cm.ScalarMappable(cmap=err_cmap, norm=err_norm)
    sm_err.set_array([])
    cb_err = fig.colorbar(sm_err, cax=err_cbar_ax,
                           orientation='horizontal',
                           extend='both')
    cb_err.outline.set_linewidth(0)
    cb_err.set_ticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    cb_err.ax.tick_params(labelsize=9.5, colors='#444444',
                           length=0, pad=4)
    for spine in cb_err.ax.spines.values():
        spine.set_visible(False)

    fig.text(err_cbar_centre, y_cbar_lbl,
              'Error (metres, prediction − truth)',
              ha='center', va='center', fontsize=10.5,
              color='#666666', style='italic')

    # ---- TITLE / SUBTITLE / CAPTION ------------------------------------
    fig.text(0.5, y_title, scene_id,
              ha='center', va='center',
              fontsize=21, weight='semibold', color='#0d0d0d')

    H, W = dsm_m.shape
    gsd = scene_meta.get('gsd') if scene_meta else None
    bbox = scene_meta.get('bbox') if scene_meta else None
    extent_parts = []

    # Lat/lon at scene centre (most identifying piece of metadata)
    latlon_str = _osgb_to_latlon_str(bbox)
    if latlon_str:
        extent_parts.append(latlon_str)

    if gsd is not None:
        extent_parts.append(
            f"{int(W * gsd):,} × {int(H * gsd):,} metres"
            f"  at  {gsd:.2f} m / pixel")
    else:
        extent_parts.append(f"{W:,} × {H:,} pixels")
    extent_parts.append(
        f"{int(v0.sum()/1e6):,} million valid pixels")

    extent_str = '   ·   '.join(extent_parts)
    fig.text(0.5, y_subtitle, extent_str,
              ha='center', va='center',
              fontsize=12, color='#888888')

    if stats is not None:
        # Caption with the key metrics — bottom of figure, monospaced
        caption_parts = []
        if 'rmse' in stats:
            caption_parts.append(f"RMSE  {stats['rmse']:.2f} m")
        if 'mae' in stats:
            caption_parts.append(f"MAE  {stats['mae']:.2f} m")
        if 'E_T1' in stats and not np.isnan(stats['E_T1']):
            caption_parts.append(f"E_T1  {stats['E_T1']*100:.1f}%")
        if 'E_T2' in stats and not np.isnan(stats['E_T2']):
            caption_parts.append(f"E_T2  {stats['E_T2']*100:.1f}%")
        caption = '     '.join(caption_parts)
        fig.text(0.5, y_caption, caption,
                  ha='center', va='center', fontsize=11,
                  color='#555555', family='monospace')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor='white', pad_inches=0.05)
    plt.close(fig)
    return out_path


# ============================================================================
# Representative scene sampling
# ============================================================================

def _select_representative_scenes(scene_files, n: int, *, log=None):
    """Pick `n` scenes spanning the geographic extent of England.

    Strategy: stratified sample by OS National Grid 100 km square (the
    first 2 chars of the LAZ filename, e.g. NX, SD, TQ). These are
    uniformly distributed over Great Britain so this produces a
    geographically-representative sample. Within each square, pick
    scenes proportionally to that square's population, but ensure
    every square gets ≥ 1 if it has any scenes.
    """
    from collections import defaultdict
    if n is None or n >= len(scene_files):
        return list(scene_files)

    groups: dict = defaultdict(list)
    for sm in scene_files:
        scene_id = sm.parent.name  # "EN_NX9212_P_..."
        body = scene_id[3:] if scene_id.startswith('EN_') else scene_id
        os_pair = body[:2].upper()
        groups[os_pair].append(sm)

    n_groups = len(groups)
    if log is not None:
        log.info(f"  representative sampling: {n} from "
                 f"{len(scene_files)} scenes across {n_groups} "
                 f"OS 100km squares")

    rng = np.random.default_rng(0)  # deterministic
    total = len(scene_files)
    allocations = {}
    for grp, files in groups.items():
        share = max(1, int(round(n * len(files) / total)))
        allocations[grp] = min(share, len(files))

    cur = sum(allocations.values())
    while cur > n:
        biggest = max(allocations, key=lambda k: allocations[k])
        if allocations[biggest] > 1:
            allocations[biggest] -= 1
            cur -= 1
        else:
            break
    while cur < n:
        space = {g: len(groups[g]) - allocations[g] for g in groups}
        if not any(v > 0 for v in space.values()):
            break
        biggest = max(space, key=lambda k: space[k])
        allocations[biggest] += 1
        cur += 1

    selected = []
    for grp in sorted(groups):
        files = sorted(groups[grp])
        idx = rng.permutation(len(files))[:allocations[grp]]
        for i in sorted(idx):
            selected.append(files[i])

    if log is not None:
        squares = sorted({s.parent.name[3:5] for s in selected
                          if s.parent.name.startswith('EN_')})
        log.info(f"  selected {len(selected)} scenes from "
                 f"{len(squares)} squares: {' '.join(squares[:30])}"
                 f"{' ...' if len(squares) > 30 else ''}")
    return selected


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--tile_dir', required=True)
    p.add_argument('--resume', required=True,
                   help="Path to a checkpoint .pt to load.")
    p.add_argument('--out_dir', required=True)
    p.add_argument('--split', default='test',
                   choices=['test', 'train', 'all'],
                   help="Which scene set to evaluate. 'all' uses both "
                        "train and test splits for England-wide stats.")
    p.add_argument('--blend_mode', default='min',
                   choices=['mean', 'min', 'max', 'linear',
                            'cosine', 'exponential'])
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--tta', type=int, default=1, choices=[1, 4, 8],
                   help="Test-time augmentation passes per tile. "
                        "1 (default) = no TTA. 4 = D2 (4 flips). "
                        "8 = D4 (4 flips + 3 rotations + 1 diag). "
                        "Costs `tta`× the inference time but kills "
                        "directional / grid artefacts.")
    p.add_argument('--init_prior', action='store_true', default=True,
                   help="Use PrioStitch prior init (paper §3.3 default).")
    p.add_argument('--no_init_prior', dest='init_prior', action='store_false')
    p.add_argument('--limit_scenes', type=int, default=None,
                   help="Process at most N scenes (for smoke tests).")
    p.add_argument('--only_scene', default=None,
                   help="Process only the scene(s) whose ID matches "
                        "(or contains as substring) the given string. "
                        "Can be a comma-separated list to redo multiple "
                        "scenes in one run, e.g. "
                        "--only_scene SX9462,SD2804,ST3262. Useful for "
                        "redoing flagged viz scenes with heavier settings "
                        "(e.g. --tta 8 --stride 64 --blend_mode linear).")
    p.add_argument('--n_scenes', type=int, default=None,
                   help="Stratified sample N scenes spanning all OS "
                        "100km squares for a geographically "
                        "representative sweep. Mutually exclusive with "
                        "--limit_scenes.")
    p.add_argument('--laz_root', default=None,
                   help="If set, classify points from each scene's raw "
                        "LAZ against the predicted DTM and report "
                        "Sithole-Vosselman E_T1/E_T2/E_tot.")
    p.add_argument('--sithole_threshold', type=float, default=0.20)
    # ----- Visualisation -----
    p.add_argument('--viz', action='store_true', default=True,
                   help="Save publication-grade triptych viz "
                        "(DSM | GT DTM | Pred DTM) per scene.")
    p.add_argument('--no_viz', dest='viz', action='store_false')
    p.add_argument('--viz_top_n', type=int, default=8,
                   help="Number of scenes to visualise by SIZE "
                        "(largest by valid pixel count).")
    p.add_argument('--viz_cliff_n', type=int, default=4,
                   help="Number of additional scenes to visualise by "
                        "CLIFFINESS (top 99.5%%ile DTM gradient — "
                        "captures sea cliffs, quarries, escarpments). "
                        "Combined with viz_top_n; duplicates removed.")
    p.add_argument('--viz_min_pixels', type=int, default=4_000_000,
                   help="Only visualise scenes with ≥ this many valid "
                        "pixels (4M ≈ 1 km² at 0.5 m/px).")
    p.add_argument('--viz_dpi', type=int, default=200,
                   help="Output resolution for triptych. 200 → "
                        "~16-megapixel image; 300 → ~36MP (large file).")
    p.add_argument('--viz_panel_inches', type=float, default=8.0,
                   help="Each panel's edge in inches at viz_dpi.")
    p.add_argument('--viz_no_hillshade', dest='viz_hillshade',
                   action='store_false', default=True,
                   help="Disable hillshade overlay (faster, flatter).")
    p.add_argument('--viz_no_inpaint', dest='viz_inpaint',
                   action='store_false', default=True,
                   help="Disable invalid-pixel inpaint (faster, "
                        "shows raw scene shape).")
    p.add_argument('--viz_vertical_exaggeration', type=float, default=4.0,
                   help="Hillshade z-scale (paper-style ≈ 3-5×).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    log = _setup_logger(out_dir)

    cfg = json.loads(Path(args.config).read_text())
    use_min_dsm = bool(cfg.get('data', {}).get('use_min_dsm', False))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GrounDiff(
        unet_kwargs=cfg['model']['unet'],
        diffusion_kwargs=cfg['model']['diffusion'],
        loss_kwargs=cfg['model'].get('loss', {}),
    ).to(device)
    d = torch.load(args.resume, map_location=device, weights_only=False)
    model.unet.load_state_dict(d['unet'] if 'unet' in d else d)
    model.eval()
    log.info(f"loaded {args.resume}; UNet={model.num_params/1e6:.1f}M params  "
             f"use_min_dsm={use_min_dsm}")

    # Discover scene files
    if args.split == 'all':
        scene_files = (sorted(
            (Path(args.tile_dir) / 'train').rglob('_scene_*.npz'))
            + sorted(
                (Path(args.tile_dir) / 'test').rglob('_scene_*.npz')))
    else:
        scene_files = sorted(
            (Path(args.tile_dir) / args.split).rglob('_scene_*.npz'))

    if args.only_scene:
        # Comma-separated list of substrings; match if scene_id contains
        # ANY of them (case-insensitive)
        needles = [s.strip().upper() for s in args.only_scene.split(',')
                    if s.strip()]
        scene_files = [
            s for s in scene_files
            if any(n in s.parent.name.upper() for n in needles)
        ]
        log.info(f"  --only_scene matched {len(scene_files)} scenes "
                 f"from {len(needles)} substring(s): "
                 f"{', '.join(needles)}")
        if not scene_files:
            log.error(f"no scenes match --only_scene '{args.only_scene}'")
            return
    elif args.n_scenes:
        scene_files = _select_representative_scenes(
            scene_files, args.n_scenes, log=log)
    elif args.limit_scenes:
        scene_files = scene_files[:args.limit_scenes]
    if args.limit_scenes and args.n_scenes:
        log.warning("both --limit_scenes and --n_scenes set; "
                    "--n_scenes takes priority")

    log.info(f"{len(scene_files)} scenes to evaluate from "
             f"split={args.split!r}  blend={args.blend_mode}  "
             f"stride={args.stride}  init_prior={args.init_prior}  "
             f"tta={args.tta}")
    if args.viz:
        log.info(f"  triptych viz: top {args.viz_top_n} largest + "
                 f"{args.viz_cliff_n} cliffiest scenes "
                 f"(≥ {args.viz_min_pixels:,} valid px each)  "
                 f"dpi={args.viz_dpi}  "
                 f"hillshade={args.viz_hillshade}  "
                 f"inpaint={args.viz_inpaint}")

    # First pass: collect scene sizes + cliff scores for viz selection.
    # Cliff score = MAX |∇DTM| over valid pixels (subsampled). This
    # specifically rewards rare, abrupt drops — sea cliffs, quarries,
    # escarpments — over smoothly-varying mountainous terrain (where
    # mean/p99 gradient would also score high). Uses DTM rather than
    # DSM so building edges don't contaminate the metric.
    scene_info: list = []
    log.info("scanning scenes (size + cliff score)...")
    for sm in scene_files:
        try:
            with np.load(str(sm), allow_pickle=True) as f:
                if 'valid' not in f.files:
                    continue
                valid = f['valid'].astype(bool)
                n_valid = int(valid.sum())
                shape = tuple(valid.shape)
                cliff_score = 0.0
                # Only score scenes that could possibly be visualised
                if (args.viz and 'dtm' in f.files
                        and n_valid >= args.viz_min_pixels):
                    dtm = f['dtm']
                    # Subsample to ~1000² for fast gradient compute,
                    # but cap step at 4 to keep cliff edges sharp
                    step = max(1, min(4, max(dtm.shape) // 1000))
                    dtm_s = dtm[::step, ::step].astype(np.float32)
                    v_s = valid[::step, ::step]
                    if v_s.any():
                        gy, gx = np.gradient(dtm_s)
                        gmag = np.sqrt(gy * gy + gx * gx)
                        # Use max gradient — discriminates abrupt drops
                        # (cliffs, quarries) from gradual mountains
                        cliff_score = float(gmag[v_s].max())
        except Exception:
            continue
        scene_info.append((sm, n_valid, shape, cliff_score))

    # Build viz set: combine largest-N + cliffiest-M (deduplicated)
    viz_set: set = set()
    viz_reasons: dict = {}  # sm -> 'size' / 'cliff' / 'both' / 'only'

    # If --only_scene is set, always viz those scenes regardless of
    # size/cliff thresholds — user explicitly asked for them.
    if args.viz and args.only_scene:
        for sm, _, _, _ in scene_info:
            viz_set.add(sm)
            viz_reasons[sm] = 'only'
        if viz_set:
            log.info(f"  selected {len(viz_set)} scenes from "
                     f"--only_scene filter")

    if args.viz and args.viz_top_n > 0:
        biggies = sorted(scene_info, key=lambda x: -x[1])
        for sm, n_valid, shape, _ in biggies:
            if n_valid < args.viz_min_pixels:
                break
            if sm in viz_set:
                continue
            viz_set.add(sm)
            viz_reasons[sm] = 'size'
            if (sum(1 for r in viz_reasons.values() if r == 'size')
                    >= args.viz_top_n):
                break
        n_size = sum(1 for r in viz_reasons.values() if r == 'size')
        log.info(f"  selected {n_size} scenes by size "
                 f"(largest by valid pixel count)")

    if args.viz and args.viz_cliff_n > 0:
        cliffy = sorted(
            [s for s in scene_info if s[1] >= args.viz_min_pixels],
            key=lambda x: -x[3])
        n_added = 0
        cliff_thresh_picked = None
        for sm, n_valid, shape, score in cliffy:
            if n_added >= args.viz_cliff_n:
                break
            if sm in viz_set:
                viz_reasons[sm] = 'both'  # already a size pick
                continue
            viz_set.add(sm)
            viz_reasons[sm] = 'cliff'
            cliff_thresh_picked = score
            n_added += 1
        if n_added > 0:
            log.info(f"  selected {n_added} cliff scenes "
                     f"(max DTM gradient ≥ {cliff_thresh_picked:.1f}"
                     f" m/px — typically natural cliffs, quarries, "
                     f"escarpments)")

    if args.viz and viz_set:
        # Show what we picked
        for sm in sorted(viz_set, key=lambda s: s.parent.name):
            scene_id = sm.parent.name
            score = next((c for s2, _, _, c in scene_info if s2 == sm), 0)
            n_valid = next((n for s2, n, _, _ in scene_info if s2 == sm), 0)
            log.info(f"    {viz_reasons[sm]:>5s}  {scene_id}  "
                     f"  {n_valid/1e6:.1f}M valid  "
                     f"max∇={score:.1f} m/px")

    g_agg = MetricAggregator()
    g_pts_TP = g_pts_FP = g_pts_FN = g_pts_TN = 0
    rows = []
    t0 = time.time()
    n_tot = len(scene_files)
    n_done = 0

    for n, sm in enumerate(scene_files, 1):
        scene_id = sm.parent.name
        try:
            s = _load_scene(sm)
        except Exception as e:
            log.warning(f"[{n}/{n_tot}] {scene_id}: load failed ({e})")
            continue
        dsm = s['dsm'].astype(np.float32)
        dtm_gt = s['dtm'].astype(np.float32)
        valid = s['valid'].astype(bool)
        stats = s['stats']
        bbox = s.get('bbox', None)
        gsd = float(s.get('gsd', cfg.get('preprocess', {}).get('gsd', 0.5))
                    if 'gsd' in s else 0.5)
        dsm_min = (s['dsm_min'].astype(np.float32)
                   if use_min_dsm and 'dsm_min' in s else None)

        if not stats.get('has_data', True):
            log.info(f"[{n}/{n_tot}] {scene_id}: no valid data, skipped")
            continue

        log.info(f"[{n}/{n_tot}] {scene_id}: "
                 f"DSM={dsm.shape}  valid={valid.mean():.1%}  "
                 f"({(time.time()-t0):.0f}s elapsed)")

        # PrioStitch handles per-tile normalisation internally; returns metres.
        pred_m, prob = priostitch_inference(
            model, dsm_full=np.zeros_like(dsm),  # unused
            dsm_metres=dsm, scene_stats=stats,
            dsm_min_metres=dsm_min,
            device=device,
            tile=cfg['model']['unet']['image_size'],
            stride=args.stride, blend_mode=args.blend_mode,
            init_prior=args.init_prior, valid_mask=valid,
            return_prob=True,
            tta=args.tta,
        )

        agg_scene = MetricAggregator()
        agg_scene.update(pred_m, dtm_gt, valid, dsm_m=dsm, prob_ground=prob)
        rs = agg_scene.result()
        g_agg.update(pred_m, dtm_gt, valid, dsm_m=dsm, prob_ground=prob)

        # ---- Per-point Sithole-Vosselman classification ----------------
        cls = dict(E_T1=float('nan'), E_T2=float('nan'), E_tot=float('nan'),
                   n_ground=0, n_nonground=0)
        if args.laz_root is not None and bbox is not None:
            laz = None
            stem = scene_id[3:] if scene_id.startswith('EN_') else scene_id
            for sub in ('Test', 'Training', '.'):
                for ext in ('.copc.laz', '.laz'):
                    cand = Path(args.laz_root) / sub / f"{stem}{ext}"
                    if cand.exists():
                        laz = cand; break
                if laz is not None:
                    break
            if laz is not None:
                pts = _read_laz_points(laz)
                if pts is not None:
                    xp, yp, zp, cp = pts
                    pred_g = classify_points_against_dtm(
                        xp, yp, zp, pred_m, valid, bbox, gsd,
                        threshold=args.sithole_threshold)
                    in_dtm = pred_g != 0.5
                    gt_g = np.isin(cp, GROUND_CLASSES)
                    cls = per_point_classification(
                        pred_g[in_dtm], gt_g[in_dtm])
                    g_pts_TP += cls['TP']; g_pts_FP += cls['FP']
                    g_pts_FN += cls['FN']; g_pts_TN += cls['TN']

        cls_source = 'per-point' if (
            args.laz_root is not None and bbox is not None
            and not np.isnan(cls['E_T1'])
        ) else 'per-cell'
        if cls_source == 'per-cell':
            cls = dict(E_T1=rs['E_T1'], E_T2=rs['E_T2'],
                       E_tot=rs['E_tot'],
                       n_ground=rs.get('n_ground', 0),
                       n_nonground=rs.get('n_nonground', 0))

        log.info(f"  RMSE={rs['rmse']:.3f}m  MAE={rs['mae']:.3f}m  "
                 f"E_T1={cls['E_T1']*100:.2f}%  "
                 f"E_T2={cls['E_T2']*100:.2f}%  "
                 f"E_tot={cls['E_tot']*100:.2f}%  "
                 f"err>0.5m={rs['err_gt_0.5']:.1%}  "
                 f"err>1.0m={rs['err_gt_1.0']:.1%}  "
                 f"[{cls_source}]")

        # Build row: regression metrics from rs, classification from cls
        # (cls takes precedence — it may be per-point Sithole-Vosselman
        # while rs is always per-cell). Drop cls keys from rs first to
        # avoid a duplicate-keyword TypeError when **rs and **cls overlap.
        rs_no_cls = {k: v for k, v in rs.items()
                      if k not in ('E_T1', 'E_T2', 'E_tot',
                                    'n_ground', 'n_nonground')}
        rows.append(dict(scene=scene_id, **rs_no_cls, **cls,
                          cls_source=cls_source))
        n_done += 1

        # ---- Triptych viz on the largest scenes -------------------------
        if sm in viz_set:
            try:
                viz_path = out_dir / 'viz' / f"{scene_id}.png"
                _render_triptych(
                    scene_id=scene_id,
                    dsm_m=dsm, dtm_gt_m=dtm_gt, pred_m=pred_m, valid=valid,
                    out_path=viz_path,
                    dpi=args.viz_dpi,
                    panel_inches=args.viz_panel_inches,
                    stats=dict(rmse=rs['rmse'], mae=rs['mae'],
                                E_T1=cls['E_T1'], E_T2=cls['E_T2']),
                    scene_meta=dict(gsd=gsd, bbox=bbox),
                    hillshade=args.viz_hillshade,
                    inpaint=args.viz_inpaint,
                    vertical_exaggeration=args.viz_vertical_exaggeration,
                )
                log.info(f"  viz → {viz_path.name} "
                         f"({viz_path.stat().st_size/1e6:.1f} MB)")
            except Exception as e:
                log.warning(f"  viz failed for {scene_id}: {e}")

    # ---- Aggregate ------------------------------------------------------
    gr = g_agg.result()
    g_n_g  = g_pts_TP + g_pts_FN
    g_n_ng = g_pts_TN + g_pts_FP
    g_n_total = g_n_g + g_n_ng
    if g_n_total > 0:
        g_e_t1 = g_pts_FP / max(g_n_ng, 1)
        g_e_t2 = g_pts_FN / max(g_n_g, 1)
        g_e_tot = g_e_t1 + g_e_t2
        cls_str = (f"per-point  E_T1={g_e_t1*100:.2f}%  "
                   f"E_T2={g_e_t2*100:.2f}%  "
                   f"E_tot={g_e_tot*100:.2f}%  "
                   f"(n_pts={g_n_total:,d})")
    else:
        g_e_t1 = gr.get('E_T1', float('nan'))
        g_e_t2 = gr.get('E_T2', float('nan'))
        g_e_tot = gr.get('E_tot', float('nan'))
        cls_str = (f"per-cell   E_T1={g_e_t1*100:.2f}%  "
                   f"E_T2={g_e_t2*100:.2f}%  E_tot={g_e_tot*100:.2f}%  "
                   f"(n_cells={gr.get('n_classified', 0):,d}; "
                   f"pass --laz_root for per-point Sithole-Vosselman)")

    log.info(
        f"\nGLOBAL  RMSE={gr['rmse']:.3f}m  MAE={gr['mae']:.3f}m  "
        f"err>0.5m={gr['err_gt_0.5']:.1%}  "
        f"err>1.0m={gr['err_gt_1.0']:.1%}  "
        f"(n_cells={gr['n_valid']:,d}, scenes={n_done})\n"
        f"GLOBAL  {cls_str}\n"
        f"        ({(time.time()-t0):.0f}s)"
    )
    gr_no_cls = {k: v for k, v in gr.items()
                  if k not in ('E_T1', 'E_T2', 'E_tot',
                                'n_ground', 'n_nonground')}
    rows.append(dict(scene='__GLOBAL__', **gr_no_cls,
                     E_T1=g_e_t1, E_T2=g_e_t2, E_tot=g_e_tot,
                     n_ground=g_n_g, n_nonground=g_n_ng,
                     cls_source='per-point' if g_n_total > 0 else 'per-cell'))

    csv_path = out_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['scene', 'rmse', 'mae', 'E_T1', 'E_T2', 'E_tot',
                      'err_gt_0.5', 'err_gt_1.0',
                      'n_valid', 'n_ground', 'n_nonground', 'cls_source']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})
    log.info(f"wrote {csv_path}")


if __name__ == '__main__':
    main()

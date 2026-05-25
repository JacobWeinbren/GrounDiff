#!/usr/bin/env python3
"""National per-tile error map for the DEFRA-2022 (England) tile set.

Reduces every scene to ONE record — its centroid + bbox (EPSG:27700,
British National Grid) joined to its metrics — and emits:

  RAW DATA (the durable, reusable artifacts):
    <out>/tile_errors.geojson   one Polygon (tile bbox) per scene, every
                                metric as a property, CRS=EPSG:27700.
                                Open in QGIS/ArcGIS, style by any field.
    <out>/tile_errors.csv       same records, flat (centroid E/N, bbox,
                                region, all metrics). For pandas/Excel.
    <out>/region_summary.csv    pooled metrics per OS 100 km region.

  RENDERS (made FROM the raw data, so they never disagree with it):
    <out>/map_rmse.png          tiles coloured by RMSE (m)
    <out>/map_etot.png          tiles coloured by E_tot (%)
    <out>/map_coverage.png      tiles coloured by DSM coverage

Design choices (and why):
  * We do NOT build a national pixel mosaic. The tiles cover only a few
    percent of England's area, so a wall-to-wall raster would be ~96 %
    empty and tens of GB for no information gain. One coloured polygon per
    tile is the honest, light, queryable representation.
  * We do NOT interpolate an error surface between tiles. With a sparse
    national sample that would invent a field unsupported by data. Tiles
    are drawn where they exist; gaps stay empty.
  * Geometry is the true bbox rectangle (not just a dot), so at national
    zoom tiles read as points but you can zoom to a region and see real
    footprints. Coordinates are BNG metres straight from raster.npz.

Source of metrics: either an existing rollup.csv (fast) or, with
--compute, it will run inference per scene first (slow; usually you
already have a rollup from run_inference_suite).

Usage:
    python -m stage2_raster.scripts.national_error_map \
        --val_root /root/work/runs/preprocessed_05m \
        --rollup   /root/work/runs/run/inference_suite/rollup.csv \
        --out      /root/work/runs/run/national_map
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

BNG_EPSG = 27700  # DEFRA National LiDAR is always British National Grid.


def _os_square_sw(letters: str):
    """SW corner (easting, northing) in BNG metres of an OS 100 km grid
    square from its two-letter code. Verified against SV=(0,0),
    SU=(400000,100000), TA=(500000,400000), NU=(400000,600000)."""
    l1 = ord(letters[0]) - ord('A')
    if l1 > 7:
        l1 -= 1                       # the OS grid omits the letter 'I'
    l2 = ord(letters[1]) - ord('A')
    if l2 > 7:
        l2 -= 1
    e = ((l1 - 2) % 5) * 500000 + (l2 % 5) * 100000
    n = (3 - (l1 // 5)) * 500000 + (4 - (l2 // 5)) * 100000
    return e, n


def _os_squares_in_extent(minx, miny, maxx, maxy):
    """Every OS 100 km square code whose footprint intersects the data
    extent, with its SW corner. Generated from the 2-letter algorithm —
    no external data needed, so the grid backdrop always works offline."""
    out = {}
    letters = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'   # note: no 'I'
    for a in letters:
        for b in letters:
            code = a + b
            e, n = _os_square_sw(code)
            if (e + 100000 >= minx and e <= maxx
                    and n + 100000 >= miny and n <= maxy):
                out[code] = (e, n)
    return out


def _draw_os_grid(ax, minx, miny, maxx, maxy):
    """Draw OS 100 km square outlines + labels as a spatial reference
    frame. The set of English land squares forms a recognisable England
    silhouette, so this doubles as the 'England outline' backdrop without
    needing a coastline file. A real coastline can be overlaid via
    --boundary."""
    from matplotlib.patches import Rectangle
    squares = _os_squares_in_extent(minx, miny, maxx, maxy)
    for code, (e, n) in squares.items():
        ax.add_patch(Rectangle((e, n), 100000, 100000, fill=False,
                                edgecolor='#b8bcc4', linewidth=0.6,
                                zorder=1))
        ax.text(e + 2000, n + 98000, code, fontsize=6, color='#9aa0a8',
                ha='left', va='top', zorder=1)


def _default_boundary() -> Path | None:
    """Bundled England outline (EPSG:27700), if present."""
    p = Path(__file__).resolve().parents[1] / 'data' / 'assets' / 'england_bng.geojson'
    return p if p.exists() else None


def _boundary_polys(boundary_path: Path):
    """Return list of exterior rings (np arrays) from a GeoJSON in BNG."""
    try:
        gj = json.loads(Path(boundary_path).read_text())
    except Exception as ex:  # noqa: BLE001
        print(f"# WARN: could not read boundary {boundary_path}: {ex}")
        return []
    feats = (gj.get('features', []) if gj.get('type') == 'FeatureCollection'
             else [gj])
    rings = []
    for ft in feats:
        geom = ft.get('geometry', ft)
        t = geom.get('type'); c = geom.get('coordinates')
        if t == 'Polygon':
            rings.extend(c)
        elif t == 'MultiPolygon':
            rings.extend(r for poly in c for r in poly)
    return [np.asarray(r) for r in rings if len(r) >= 4]


def _draw_boundary(ax, boundary_path: Path, *, fill=True):
    """Draw the England landmass: soft filled polygons + a crisp coast
    line, so tiles sit on a recognisable country rather than empty grid."""
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection
    rings = _boundary_polys(boundary_path)
    if not rings:
        return
    if fill:
        patches = [MplPoly(r, closed=True) for r in rings if r.ndim == 2]
        ax.add_collection(PatchCollection(
            patches, facecolors='#e9ecf1', edgecolors='#b0b6c0',
            linewidths=0.4, zorder=1))
    for r in rings:
        if r.ndim == 2 and r.shape[0] > 1:
            ax.plot(r[:, 0], r[:, 1], color='#8a909a', linewidth=0.5,
                    zorder=2)


def _region_of(scene: str) -> str:
    """OS 100 km grid prefix (first two letters of the grid token), e.g.
    'SU60ne_P_...' -> 'SU', 'P_12345_SU6570_...' -> 'SU'. Falls back to
    the first 2 uppercase letters found."""
    m = re.match(r'^([A-Z]{2})\d', scene)
    if m:
        return m.group(1)
    m = re.search(r'([A-Z]{2})\d{2,}', scene)
    return m.group(1) if m else '??'


def _read_bbox(raster_path: Path):
    """Return (xmin, ymin, xmax, ymax) in BNG metres, or None."""
    try:
        with np.load(str(raster_path)) as z:
            if 'bbox' not in z.files:
                return None
            return tuple(float(v) for v in z['bbox'])
    except Exception:
        return None


def load_rollup(rollup_csv: Path) -> dict[str, dict]:
    """scene -> metrics dict (numbers coerced to float where possible)."""
    rows = {}
    with open(rollup_csv) as f:
        for r in csv.DictReader(f):
            name = r.get('scene')
            if not name:
                continue
            rec = {}
            for k, v in r.items():
                if k == 'scene':
                    continue
                try:
                    rec[k] = float(v)
                except (TypeError, ValueError):
                    rec[k] = v
            rows[name] = rec
    return rows


DEFAULT_COLOR_METRICS = ['rmse_ground_m', 'rmse_nonground_m', 'rmse_m',
                         'e_t1_pct', 'e_t2_pct']

_MAP_META = {
    'rmse_ground_m':    ('RMSE — ground (m)',            'rmse_ground', 'rmse'),
    'rmse_nonground_m': ('RMSE — non-ground (m)',        'rmse_nonground', 'rmse'),
    'rmse_m':           ('RMSE — total (m)',             'rmse_total', 'rmse'),
    'e_t1_pct':         ('E_T1 — retained non-ground (%)', 'e_t1', 'pct'),
    'e_t2_pct':         ('E_T2 — removed ground (%)',     'e_t2', 'pct'),
    'e_tot_pct':        ('E_tot (%)',                     'e_tot', 'pct'),
    'frac_within_return': ('DSM coverage',               'coverage', 'cov'),
}


def make_record(scene: str, bbox, metrics: dict) -> dict:
    """One map record: footprint (BNG metres) + region + metrics. Shared by
    the standalone map CLI and the live in-loop preview so both build
    identical records."""
    xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    rec = dict(scene=scene, region=_region_of(scene),
               xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
               cx=0.5 * (xmin + xmax), cy=0.5 * (ymin + ymax),
               area_km2=(xmax - xmin) * (ymax - ymin) / 1e6)
    rec.update(metrics)
    nv = metrics.get('n_cells_valid_gt')
    nr = metrics.get('n_cells_with_return')
    if nv and nr is not None:
        rec['frac_within_return'] = nr / nv
    return rec


def write_raw(records: list[dict], out: Path):
    """Write tile_errors.csv + tile_errors.geojson (EPSG:27700) +
    region_summary.csv from the records list."""
    out.mkdir(parents=True, exist_ok=True)
    all_keys = []
    for r in records:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with open(out / 'tile_errors.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in records:
            w.writerow(r)

    feats = []
    for r in records:
        ring = [[r['xmin'], r['ymin']], [r['xmax'], r['ymin']],
                [r['xmax'], r['ymax']], [r['xmin'], r['ymax']],
                [r['xmin'], r['ymin']]]
        props = {k: v for k, v in r.items()
                 if k not in ('xmin', 'ymin', 'xmax', 'ymax')}
        feats.append(dict(type='Feature',
                          geometry=dict(type='Polygon', coordinates=[ring]),
                          properties=props))
    gj = dict(type='FeatureCollection',
              crs=dict(type='name', properties=dict(
                  name=f'urn:ogc:def:crs:EPSG::{BNG_EPSG}')),
              features=feats)
    (out / 'tile_errors.geojson').write_text(json.dumps(gj))

    regs = {}
    for r in records:
        regs.setdefault(r['region'], []).append(r)

    def _pooled(rs, num, den):
        n = sum(rr.get(num, 0) or 0 for rr in rs)
        d = sum(rr.get(den, 0) or 0 for rr in rs)
        return (100.0 * n / d) if d else float('nan')

    with open(out / 'region_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['region', 'n_tiles', 'area_km2', 'rmse_m_median',
                    'e_tot_pct_pooled', 'e_t1_pct_pooled', 'e_t2_pct_pooled',
                    'coverage_median'])
        for reg in sorted(regs):
            rs = regs[reg]
            rmse = [rr['rmse_m'] for rr in rs
                    if isinstance(rr.get('rmse_m'), float)
                    and not np.isnan(rr['rmse_m'])]
            cov = [rr['frac_within_return'] for rr in rs
                   if isinstance(rr.get('frac_within_return'), float)]
            has = all('n_fp' in rr for rr in rs)
            w.writerow([
                reg, len(rs), f"{sum(rr['area_km2'] for rr in rs):.2f}",
                f"{np.median(rmse):.4f}" if rmse else 'nan',
                f"{_pooled(rs, 'n_fp', 'n_cells_cls_valid'):.3f}" if has else 'nan',
                f"{_pooled(rs, 'n_fp', 'n_gt_non_ground'):.3f}" if has else 'nan',
                f"{_pooled(rs, 'n_fn', 'n_gt_ground'):.3f}" if has else 'nan',
                f"{np.median(cov):.3f}" if cov else 'nan'])


def render_maps(records: list[dict], out: Path, *,
                color_metrics=None, boundary: Path | None = None,
                cmap: str = 'magma', rmse_clip: float = 2.0,
                pct_clip: float = 0.0, dpi: int = 220,
                title_note: str = '', quiet: bool = False) -> list[Path]:
    """Render one PNG per metric: tiles shaded PER FILE at their true
    footprint, over an OS-grid backdrop. Returns the written paths.
    Importable so the inference loop can call it for a live preview."""
    out.mkdir(parents=True, exist_ok=True)
    color_metrics = color_metrics or DEFAULT_COLOR_METRICS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    if not records:
        return []
    # Frame on England (from the boundary) so the country is always fully
    # shown, not just cropped to wherever tiles happen to fall.
    if boundary is None:
        boundary = _default_boundary()
    b_rings = _boundary_polys(boundary) if boundary is not None else []
    if b_rings:
        bx = np.concatenate([r[:, 0] for r in b_rings])
        by = np.concatenate([r[:, 1] for r in b_rings])
        minx, maxx = float(bx.min()), float(bx.max())
        miny, maxy = float(by.min()), float(by.max())
    else:
        minx = min(r['xmin'] for r in records); maxx = max(r['xmax'] for r in records)
        miny = min(r['ymin'] for r in records); maxy = max(r['ymax'] for r in records)
    total_area = sum(r['area_km2'] for r in records)
    span_x = max(maxx - minx, 1.0); span_y = max(maxy - miny, 1.0)
    aspect = span_y / span_x
    # Tiles are drawn at their TRUE footprint (sizes vary in the real data).
    # We only nudge genuinely sub-pixel tiles up to a small floor so they
    # don't vanish — but the floor is ~1/700 of the frame (a few px), small
    # enough that real size differences remain visible rather than being
    # flattened to identical markers.
    min_marker_m = max(span_x, span_y) / 700.0

    written = []
    for metric in color_metrics:
        vals = [r[metric] for r in records
                if isinstance(r.get(metric), (int, float))
                and not (isinstance(r.get(metric), float) and np.isnan(r[metric]))]
        if not vals:
            if not quiet:
                print(f"# skip render '{metric}': no values yet")
            continue
        finite = np.asarray(vals, dtype=float)
        label, suffix, kind = _MAP_META.get(metric, (metric, metric, 'pct'))
        if kind == 'rmse':
            vmin, vmax, cm = 0.0, rmse_clip, cmap
        elif kind == 'cov':
            vmin, vmax, cm = 0.0, 1.0, 'cividis'
        else:
            vmax = pct_clip if pct_clip > 0 else float(np.percentile(finite, 95))
            vmin, vmax, cm = 0.0, max(vmax, 1e-6), cmap

        fig = plt.figure(figsize=(10.0, 10.0 * aspect + 1.4), dpi=dpi)
        ax = fig.add_axes([0.06, 0.07, 0.86, 0.87])
        ax.set_facecolor('#f7f8fa')
        cmap_o = plt.get_cmap(cm).copy()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        # Backdrop: England land fill only (no grid overlay).
        if boundary is not None:
            _draw_boundary(ax, boundary, fill=True)
        patches, colors = [], []
        for r in records:
            v = r.get(metric)
            if not isinstance(v, (int, float)) or (
                    isinstance(v, float) and np.isnan(v)):
                continue
            # True footprint; only floor a dimension if it's sub-pixel.
            tw = r['xmax'] - r['xmin']; th = r['ymax'] - r['ymin']
            w_ = tw if tw >= min_marker_m else min_marker_m
            h_ = th if th >= min_marker_m else min_marker_m
            patches.append(Rectangle((r['cx'] - w_ / 2, r['cy'] - h_ / 2), w_, h_))
            colors.append(cmap_o(min(norm(v), 1.0)))
        ax.add_collection(PatchCollection(patches, facecolors=colors,
                                          edgecolors='none', zorder=3))
        ax.set_xlim(minx - span_x * 0.03, maxx + span_x * 0.03)
        ax.set_ylim(miny - span_y * 0.03, maxy + span_y * 0.03)
        ax.set_aspect('equal')
        ax.set_xlabel('Easting (m, EPSG:27700 — British National Grid)', fontsize=9)
        ax.set_ylabel('Northing (m)', fontsize=9)
        note = f' · {title_note}' if title_note else ''
        ax.set_title(f'DEFRA-2022 England · per-tile {label}\n'
                     f'{len(patches)} tiles · {total_area:,.0f} km$^2$ · '
                     f'coloured per file{note}', fontsize=11)
        sm = plt.cm.ScalarMappable(cmap=cmap_o, norm=norm); sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.02,
                     extend='max').set_label(label)
        png = out / f'map_{suffix}.png'
        fig.savefig(png, dpi=dpi, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        written.append(png)
        if not quiet:
            print(f"# wrote {png}  (vmax={vmax:.3g})")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--val_root', required=True, type=Path,
                    help='dir of <scene>/raster.npz (for bboxes)')
    ap.add_argument('--rollup', required=True, type=Path,
                    help='rollup.csv from run_inference_suite / eval')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--color-metrics', nargs='+',
                    default=['rmse_ground_m', 'rmse_nonground_m', 'rmse_m',
                             'e_t1_pct', 'e_t2_pct'],
                    help='metric columns to render (skips any not in rollup). '
                         'Default: ground/non-ground/total RMSE, E_T1, E_T2.')
    ap.add_argument('--boundary', type=Path, default=None,
                    help='optional GeoJSON boundary (EPSG:27700) drawn as an '
                         'outline over the OS-grid backdrop, e.g. an England '
                         'coastline. Omit to use the OS grid alone.')
    ap.add_argument('--cmap', type=str, default='magma',
                    help='matplotlib colormap for error maps. Default magma '
                         '(perceptually uniform + colourblind-friendly; '
                         'dark=low error, bright=high). cividis/viridis also '
                         'CVD-safe.')
    ap.add_argument('--rmse-clip', type=float, default=2.0,
                    help='upper clip for RMSE colour scale (m); outliers '
                         'still recorded in raw data, just saturate in PNG')
    ap.add_argument('--pct-clip', type=float, default=0.0,
                    help='upper clip for percentage maps (0 = use 95th pct)')
    ap.add_argument('--dpi', type=int, default=220)
    ap.add_argument('--no-png', action='store_true',
                    help='write only the raw GeoJSON/CSV, skip rendering')
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    metrics = load_rollup(args.rollup)
    print(f"# rollup: {len(metrics)} scenes with metrics")

    # ---- join bboxes ----
    scenes = sorted(p.parent.name for p in args.val_root.glob('*/raster.npz'))
    print(f"# val_root: {len(scenes)} scenes with raster.npz")

    records = []
    n_no_bbox = n_no_metric = 0
    for name in scenes:
        bbox = _read_bbox(args.val_root / name / 'raster.npz')
        if bbox is None:
            n_no_bbox += 1; continue
        m = metrics.get(name)
        if m is None:
            n_no_metric += 1; continue
        records.append(make_record(name, bbox, m))

    print(f"# joined {len(records)} scenes "
          f"({n_no_bbox} missing bbox, {n_no_metric} missing metrics)")
    if not records:
        raise SystemExit("no scenes with both bbox and metrics")

    total_area = sum(r['area_km2'] for r in records)
    minx = min(r['xmin'] for r in records); maxx = max(r['xmax'] for r in records)
    miny = min(r['ymin'] for r in records); maxy = max(r['ymax'] for r in records)
    print(f"# total covered area = {total_area:,.1f} km^2 across "
          f"{len(set(r['region'] for r in records))} OS regions")
    print(f"# BNG extent: E [{minx:,.0f}, {maxx:,.0f}]  N [{miny:,.0f}, {maxy:,.0f}]")

    write_raw(records, out)
    print(f"# wrote tile_errors.csv / .geojson / region_summary.csv")

    if args.no_png:
        print("# --no-png: skipping renders"); return
    render_maps(records, out, color_metrics=args.color_metrics,
                boundary=args.boundary, cmap=args.cmap,
                rmse_clip=args.rmse_clip, pct_clip=args.pct_clip, dpi=args.dpi)
    print(f"\n# done -> {out}")
    print(f"#   raw:    tile_errors.geojson (QGIS), tile_errors.csv, region_summary.csv")
    print(f"#   render: map_*.png")


if __name__ == '__main__':
    main()

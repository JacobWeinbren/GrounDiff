#!/usr/bin/env python3
"""Build a comprehensive CSV inventory of all DEFRA LIDAR scenes.

Walks the raw LAZ dir and the preprocessed tile dir, joining them by
scene ID. Writes one row per scene with metadata useful for future
reference: file paths, split assignment, dimensions, bbox, lat/lon
centre, point count, tile count, etc.

Usage:
    # Default paths (matches the standard layout)
    python scripts/inventory.py

    # Custom paths + output
    python scripts/inventory.py \\
        --laz_root /data/data/england_raw \\
        --tile_dir /data/england_tiles \\
        --out inventory.csv

    # Skip LAZ point counts (faster — just file sizes)
    python scripts/inventory.py --no_point_count

Schema (one row per scene_id):
    scene_id              EN_NX9214_P_12706_20220902_20220902
    os_grid_pair          NX
    os_grid_ref           NX9214
    laz_path              /data/data/england_raw/NX9214_P_...
    laz_exists            True / False
    laz_size_mb           123.4
    laz_point_count       4567890         (or '' if --no_point_count)
    laz_readable          True / False    (False if corrupted)
    scene_marker_path     /data/england_tiles/train/EN_NX9214_.../_scene_*.npz
    scene_marker_exists   True / False
    split                 train / test / none
    dsm_height            1794
    dsm_width             981
    n_valid               263584
    valid_fraction        0.150
    n_tiles               12              (.npz tile files, excluding scene marker)
    bbox_xmin             392000.0        (OSGB easting)
    bbox_ymin             214000.0        (OSGB northing)
    bbox_xmax             393000.0
    bbox_ymax             215000.0
    centre_lat            54.5257         (WGS84)
    centre_lon            -2.1174
    gsd                   0.50            (m/px)
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def _read_laz_header(path: Path, *, get_point_count: bool = True):
    """Read just the LAZ header — fast, doesn't decompress points."""
    try:
        import laspy
    except ImportError:
        return None
    try:
        with laspy.open(str(path)) as r:
            n = int(r.header.point_count) if get_point_count else None
            return dict(point_count=n, readable=True)
    except Exception:
        return dict(point_count=None, readable=False)


def _osgb_to_latlon(easting: float, northing: float):
    """OSGB (EPSG:27700) → WGS84 (lat, lon). Returns (None, None) on
    failure (pyproj not installed, etc.)."""
    try:
        from pyproj import Transformer
        if not hasattr(_osgb_to_latlon, '_t'):
            _osgb_to_latlon._t = Transformer.from_crs(
                'EPSG:27700', 'EPSG:4326', always_xy=True)
        lon, lat = _osgb_to_latlon._t.transform(easting, northing)
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return None, None
        return float(lat), float(lon)
    except Exception:
        return None, None


def _scene_id_from_laz(laz_path: Path) -> str:
    """Convert /path/to/NX9214_P_12706_..._.copc.laz → EN_NX9214_P_12706_..."""
    name = laz_path.name
    if name.endswith('.copc.laz'):
        stem = name[:-len('.copc.laz')]
    elif name.endswith('.laz'):
        stem = name[:-len('.laz')]
    else:
        stem = laz_path.stem
    return f"EN_{stem}"


def _os_grid_pair(scene_id: str) -> str:
    """Extract OS National Grid letter pair (NX, SD, TQ, ...) from scene id."""
    body = scene_id[3:] if scene_id.startswith('EN_') else scene_id
    return body[:2].upper() if len(body) >= 2 else ''


def _os_grid_ref(scene_id: str) -> str:
    """Extract the full OS reference (e.g. NX9214) from scene_id."""
    body = scene_id[3:] if scene_id.startswith('EN_') else scene_id
    # Format is <2 letters><4 digits>_<rest>
    if '_' in body:
        ref = body.split('_', 1)[0]
    else:
        ref = body[:6]
    return ref


def _read_marker_meta(marker_path: Path) -> dict:
    """Pull dims, bbox, gsd, valid_fraction etc. from a preprocessed
    scene marker. Returns {} on failure."""
    try:
        with np.load(str(marker_path), allow_pickle=True) as f:
            files = set(f.files)
            out = {}
            if 'valid' in files:
                v = f['valid'].astype(bool)
                out['dsm_height'] = int(v.shape[0])
                out['dsm_width'] = int(v.shape[1])
                out['n_valid'] = int(v.sum())
                out['valid_fraction'] = (
                    out['n_valid'] / float(v.size) if v.size else 0.0)
            if 'bbox' in files:
                bbox = f['bbox']
                if hasattr(bbox, 'tolist'):
                    bbox = bbox.tolist()
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    out['bbox_xmin'] = float(bbox[0])
                    out['bbox_ymin'] = float(bbox[1])
                    out['bbox_xmax'] = float(bbox[2])
                    out['bbox_ymax'] = float(bbox[3])
            if 'gsd' in files:
                out['gsd'] = float(f['gsd'])
            return out
    except Exception:
        return {}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--laz_root', default='/data/data/england_raw',
                   help="Directory containing the raw .copc.laz files.")
    p.add_argument('--tile_dir', default='/data/england_tiles',
                   help="Directory with train/ and test/ subdirs of "
                        "preprocessed scenes.")
    p.add_argument('--out', default='scene_inventory.csv',
                   help="Output CSV path.")
    p.add_argument('--no_point_count', dest='get_point_count',
                   action='store_false', default=True,
                   help="Skip reading LAZ headers for point counts "
                        "(faster — relies on filesystem stat only).")
    p.add_argument('--quiet', action='store_true',
                   help="Suppress per-scene log output.")
    args = p.parse_args()

    laz_root = Path(args.laz_root)
    tile_dir = Path(args.tile_dir)
    out_path = Path(args.out)

    # ---------- 1. Index preprocessed scene markers ----------------------
    print(f"Scanning preprocessed scenes under {tile_dir}/...")
    markers: dict = {}   # scene_id -> (split, marker_path)
    for split in ('train', 'test'):
        split_dir = tile_dir / split
        if not split_dir.exists():
            continue
        for sm in split_dir.rglob('_scene_*.npz'):
            scene_id = sm.parent.name
            markers[scene_id] = (split, sm)
    print(f"  found {len(markers)} preprocessed scenes "
          f"({sum(1 for v in markers.values() if v[0] == 'train')} train, "
          f"{sum(1 for v in markers.values() if v[0] == 'test')} test)")

    # ---------- 2. Index raw LAZ files -----------------------------------
    print(f"Scanning raw LAZ files under {laz_root}/...")
    laz_files: dict = {}
    if laz_root.exists():
        for laz in laz_root.rglob('*.copc.laz'):
            scene_id = _scene_id_from_laz(laz)
            laz_files[scene_id] = laz
        # Also pick up plain .laz (if any)
        for laz in laz_root.rglob('*.laz'):
            if str(laz).endswith('.copc.laz'):
                continue
            scene_id = _scene_id_from_laz(laz)
            laz_files.setdefault(scene_id, laz)
    print(f"  found {len(laz_files)} raw LAZ files")

    # ---------- 3. Union of scene IDs ------------------------------------
    all_scenes = sorted(set(markers.keys()) | set(laz_files.keys()))
    print(f"  total {len(all_scenes)} unique scenes to inventory\n")

    # ---------- 4. Build rows --------------------------------------------
    fieldnames = [
        'scene_id', 'os_grid_pair', 'os_grid_ref',
        'laz_path', 'laz_exists', 'laz_size_mb',
        'laz_point_count', 'laz_readable',
        'scene_marker_path', 'scene_marker_exists',
        'split', 'dsm_height', 'dsm_width',
        'n_valid', 'valid_fraction', 'n_tiles',
        'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
        'centre_lat', 'centre_lon', 'gsd',
    ]

    rows = []
    n_total = len(all_scenes)
    for i, scene_id in enumerate(all_scenes, 1):
        if not args.quiet and i % 50 == 0:
            print(f"  [{i}/{n_total}] {scene_id}")

        row = {k: '' for k in fieldnames}
        row['scene_id'] = scene_id
        row['os_grid_pair'] = _os_grid_pair(scene_id)
        row['os_grid_ref'] = _os_grid_ref(scene_id)

        # LAZ file
        laz_path = laz_files.get(scene_id)
        if laz_path is not None:
            row['laz_path'] = str(laz_path)
            row['laz_exists'] = laz_path.exists()
            if laz_path.exists():
                row['laz_size_mb'] = round(
                    laz_path.stat().st_size / (1024 ** 2), 2)
                if args.get_point_count:
                    hdr = _read_laz_header(laz_path,
                                           get_point_count=True)
                    if hdr is not None:
                        row['laz_point_count'] = (
                            hdr['point_count']
                            if hdr['point_count'] is not None else '')
                        row['laz_readable'] = hdr['readable']
                else:
                    row['laz_readable'] = ''  # not checked
        else:
            row['laz_exists'] = False

        # Scene marker / split
        marker_info = markers.get(scene_id)
        if marker_info is not None:
            split, marker_path = marker_info
            row['scene_marker_path'] = str(marker_path)
            row['scene_marker_exists'] = marker_path.exists()
            row['split'] = split

            meta = _read_marker_meta(marker_path)
            for k in ('dsm_height', 'dsm_width', 'n_valid',
                       'valid_fraction', 'bbox_xmin', 'bbox_ymin',
                       'bbox_xmax', 'bbox_ymax', 'gsd'):
                if k in meta:
                    row[k] = meta[k]

            # Count tile npzs (tiles are <scene>/<scene>_NNNN_NNNN.npz,
            # exclude the marker which starts with `_scene_`)
            scene_dir = marker_path.parent
            tile_count = sum(
                1 for p in scene_dir.glob('*.npz')
                if not p.name.startswith('_scene_'))
            row['n_tiles'] = tile_count

            # Lat/lon centre from bbox
            if all(k in meta for k in
                   ('bbox_xmin', 'bbox_ymin',
                    'bbox_xmax', 'bbox_ymax')):
                cx = (meta['bbox_xmin'] + meta['bbox_xmax']) / 2
                cy = (meta['bbox_ymin'] + meta['bbox_ymax']) / 2
                lat, lon = _osgb_to_latlon(cx, cy)
                if lat is not None:
                    row['centre_lat'] = round(lat, 6)
                    row['centre_lon'] = round(lon, 6)
        else:
            row['scene_marker_exists'] = False
            row['split'] = 'none'  # raw LAZ exists but never preprocessed

        # Round floats for cleaner CSV
        for k in ('valid_fraction',):
            if isinstance(row[k], float):
                row[k] = round(row[k], 4)

        rows.append(row)

    # ---------- 5. Write CSV ---------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nwrote {out_path}  ({len(rows)} rows)")

    # ---------- 6. Summary -----------------------------------------------
    n_train = sum(1 for r in rows if r['split'] == 'train')
    n_test  = sum(1 for r in rows if r['split'] == 'test')
    n_none  = sum(1 for r in rows if r['split'] == 'none')
    n_laz   = sum(1 for r in rows if r['laz_exists'])
    n_unread = sum(1 for r in rows
                    if r['laz_exists'] and r['laz_readable'] is False)
    print(f"\nSummary:")
    print(f"  Total scenes:           {len(rows)}")
    print(f"  With raw LAZ:           {n_laz}")
    print(f"  Unreadable LAZ:         {n_unread}")
    print(f"  Preprocessed (train):   {n_train}")
    print(f"  Preprocessed (test):    {n_test}")
    print(f"  Raw LAZ but unprocessed: {n_none}")

    # Distribution by OS grid square
    from collections import Counter
    by_pair = Counter(r['os_grid_pair'] for r in rows)
    print(f"\n  Distribution across OS 100km squares "
          f"({len(by_pair)} squares):")
    for pair, count in sorted(by_pair.items()):
        print(f"    {pair}: {count}")


if __name__ == '__main__':
    main()

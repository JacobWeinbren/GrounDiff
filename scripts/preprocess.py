"""LAZ → (DSM, DTM) raster tiles for GrounDiff training.

Per scene LAZ:
  1. Read points + classifications
  2. Rasterise DSM via max-z per cell  (paper §7.1)
  3. Rasterise DTM via TIN-interpolation of ground-class returns
     (cls=2 ∪ cls=9 — see DECISIONS.md for the cls=9/water deviation)
  4. Apply paper Eq.15 joint min-max normalisation in the tile frame
  5. Slice into 256×256 tiles, save each as compressed npz

Output structure:
    {tile_dir}/{split}/{scene_id}/{scene_id}_{ti:04d}_{tj:04d}.npz
        keys: dsm_norm[H,W] f16, dtm_norm[H,W] f16, valid_mask[H,W] f32,
              stats={vmin,vmax,has_data}, tile_origin=(i,j)
    {tile_dir}/{split}/{scene_id}/_scene_{scene_id}.npz
        keys: dsm[H,W] f16, dtm[H,W] f16, valid[H,W] f32,
              bbox=(x0,y0,x1,y1), gsd=float

The per-scene `_scene_*.npz` is for PrioStitch inference (we need the
full DSM, not tiles) and qualitative scene-level visualisation.

Process pool note:
  laspy can segfault uncatchably on malformed COPC LAZ files (we hit
  this on ~6 DEFRA scenes). ProcessPoolExecutor deadlocks when a
  worker dies. Mitigation: pre-validate file size (<4KB skip) and try
  a header-only `laspy.open()` before pool. Workers also reraise as
  None-on-failure rather than crashing the pool.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Local
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.normalize import joint_minmax_normalise   # noqa: E402


# ---- LAZ I/O + rasterisation ------------------------------------------

GROUND_CLASSES = (2, 9)   # LAS class codes: 2=ground, 9=water


def _read_laz(path: Path):
    """Load (x, y, z, classification) from a LAZ. Returns None on
    corrupted files. Light-weight — no copy of all attributes."""
    try:
        import laspy
    except ImportError as e:
        raise SystemExit(
            "laspy is required for preprocess. "
            "Install with: pip install laspy[lazrs]") from e

    if path.stat().st_size < 4096:
        return None
    try:
        with laspy.open(str(path)) as r:
            n = r.header.point_count
            if n < 100:
                return None
            las = r.read()
        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        z = np.asarray(las.z, dtype=np.float64)
        c = np.asarray(las.classification, dtype=np.uint8)
        return x, y, z, c
    except Exception:
        return None


def _bbox_from_xy(x, y, gsd):
    """Snap a point cloud bbox to a regular GSD grid for raster output.
    Returns (x0, y0, x1, y1, W, H)."""
    x0 = float(np.floor(x.min() / gsd) * gsd)
    y0 = float(np.floor(y.min() / gsd) * gsd)
    x1 = float(np.ceil (x.max() / gsd) * gsd)
    y1 = float(np.ceil (y.max() / gsd) * gsd)
    W = max(1, int(round((x1 - x0) / gsd)))
    H = max(1, int(round((y1 - y0) / gsd)))
    return x0, y0, x1, y1, W, H


def _max_z_raster(x, y, z, x0, y0, W, H, gsd):
    """DSM = max-z per grid cell (paper §7.1).

    Uses np.maximum.at for vectorised in-place reduction. Fastest
    pure-numpy path for this size of point cloud (a few M points).
    """
    j = np.clip(((x - x0) / gsd).astype(np.int64), 0, W - 1)
    i = np.clip(((y1_for(y0, H, gsd) - y) / gsd).astype(np.int64), 0, H - 1)
    dsm = np.full((H, W), -np.inf, dtype=np.float64)
    np.maximum.at(dsm, (i, j), z)
    valid = np.isfinite(dsm)
    dsm = np.where(valid, dsm, 0.0).astype(np.float32)
    return dsm, valid


def _min_z_raster(x, y, z, x0, y0, W, H, gsd):
    """DSM_min = min-z per grid cell.

    For canopy/vegetation cells this is often the ground return that
    pierced through gaps in the canopy — a strong implicit prior for
    the DTM. For roofs, water, and bare ground it's just the same as
    max-z (single return per cell). Adding this as a 2nd conditioning
    channel is a paper deviation but a strong inductive bias for
    forested terrain.
    """
    j = np.clip(((x - x0) / gsd).astype(np.int64), 0, W - 1)
    i = np.clip(((y1_for(y0, H, gsd) - y) / gsd).astype(np.int64), 0, H - 1)
    dsm_min = np.full((H, W), np.inf, dtype=np.float64)
    np.minimum.at(dsm_min, (i, j), z)
    valid = np.isfinite(dsm_min)
    dsm_min = np.where(valid, dsm_min, 0.0).astype(np.float32)
    return dsm_min, valid


def y1_for(y0, H, gsd):
    """Top-edge y of an HxW raster anchored at (x0, y0) bottom-left in
    cartographic convention. Image rows are top-down so row 0 == y1."""
    return y0 + H * gsd


def _tin_dtm(x_g, y_g, z_g, x0, y0, W, H, gsd):
    """DTM = TIN-interpolate ground/water returns onto the GSD grid.

    Implementation: scipy LinearNDInterpolator over the (x, y, z) of
    classified ground+water points. Cells outside the convex hull of
    the ground points are returned as NaN; downstream we mark them
    invalid in the joint mask.
    """
    if x_g.size < 3:
        return (np.zeros((H, W), dtype=np.float32),
                np.zeros((H, W), dtype=bool))
    try:
        from scipy.interpolate import LinearNDInterpolator
    except ImportError as e:
        raise SystemExit("scipy is required for TIN DTM. "
                         "Install with: pip install scipy") from e

    # Output grid centres (geo coords). Image row 0 == y1 (top).
    y1 = y1_for(y0, H, gsd)
    xs = x0 + (np.arange(W) + 0.5) * gsd
    ys = y1 - (np.arange(H) + 0.5) * gsd
    XX, YY = np.meshgrid(xs, ys)

    interp = LinearNDInterpolator(np.column_stack([x_g, y_g]), z_g,
                                  fill_value=np.nan)
    Z = interp(XX, YY).astype(np.float32)
    valid = np.isfinite(Z)
    Z = np.where(valid, Z, 0.0)
    return Z, valid


def _slice_into_tiles(dsm_norm, dtm_norm, valid, *, tile=256, stride=None,
                      min_valid_frac=0.05):
    """Slice the per-scene rasters into non-overlapping (or strided)
    tile×tile patches. Returns list of (i, j, dsm_t, dtm_t, valid_t).

    `min_valid_frac` skips tiles with too few valid pixels — these are
    almost-empty edges where the diffusion would have nothing to learn.
    """
    H, W = dsm_norm.shape
    s = int(stride or tile)
    out = []
    for i in range(0, H, s):
        for j in range(0, W, s):
            i1 = min(i + tile, H)
            j1 = min(j + tile, W)
            v = valid[i:i1, j:j1]
            if v.size == 0:
                continue
            if v.mean() < min_valid_frac:
                continue
            dsm_t = np.zeros((tile, tile), dtype=np.float32)
            dtm_t = np.zeros((tile, tile), dtype=np.float32)
            val_t = np.zeros((tile, tile), dtype=np.float32)
            dsm_t[:i1 - i, :j1 - j] = dsm_norm[i:i1, j:j1]
            dtm_t[:i1 - i, :j1 - j] = dtm_norm[i:i1, j:j1]
            val_t[:i1 - i, :j1 - j] = v.astype(np.float32)
            out.append((i, j, dsm_t, dtm_t, val_t))
    return out


def _slice_into_tiles_with_min(dsm_norm, dsm_min_norm, dtm_norm, valid,
                                *, tile=256, stride=None,
                                min_valid_frac=0.05):
    """Same as `_slice_into_tiles` but additionally carries the min-z
    DSM raster. Returns list of (i, j, dsm_t, dsm_min_t, dtm_t, valid_t).
    """
    H, W = dsm_norm.shape
    s = int(stride or tile)
    out = []
    for i in range(0, H, s):
        for j in range(0, W, s):
            i1 = min(i + tile, H)
            j1 = min(j + tile, W)
            v = valid[i:i1, j:j1]
            if v.size == 0:
                continue
            if v.mean() < min_valid_frac:
                continue
            dsm_t = np.zeros((tile, tile), dtype=np.float32)
            dsm_min_t = np.zeros((tile, tile), dtype=np.float32)
            dtm_t = np.zeros((tile, tile), dtype=np.float32)
            val_t = np.zeros((tile, tile), dtype=np.float32)
            dsm_t[:i1 - i, :j1 - j] = dsm_norm[i:i1, j:j1]
            dsm_min_t[:i1 - i, :j1 - j] = dsm_min_norm[i:i1, j:j1]
            dtm_t[:i1 - i, :j1 - j] = dtm_norm[i:i1, j:j1]
            val_t[:i1 - i, :j1 - j] = v.astype(np.float32)
            out.append((i, j, dsm_t, dsm_min_t, dtm_t, val_t))
    return out


# ---- Scene-level worker ------------------------------------------------

def _process_scene(args):
    """Single-scene worker. Returns (scene_id, n_tiles, error_or_None)."""
    if len(args) == 7:
        (laz_path, out_dir_str, gsd, tile, stride, min_valid_frac,
         scene_prefix) = args
    else:
        # Backward compat: 6-arg form (prefix defaults to '')
        laz_path, out_dir_str, gsd, tile, stride, min_valid_frac = args
        scene_prefix = ''
    laz = Path(laz_path)
    out_dir = Path(out_dir_str)
    # Strip both .copc and .laz suffixes for clean scene_id
    stem = laz.stem
    if stem.endswith('.copc'):
        stem = stem[:-5]
    scene_id = f"{scene_prefix}{stem}"
    scene_out = out_dir / scene_id
    try:
        # Idempotent: skip if scene marker exists
        scene_marker = scene_out / f"_scene_{scene_id}.npz"
        if scene_marker.exists():
            n_tiles = sum(1 for p in scene_out.glob('*.npz')
                          if not p.name.startswith('_scene_'))
            return scene_id, n_tiles, None

        rec = _read_laz(laz)
        if rec is None:
            return scene_id, 0, "unreadable LAZ"
        x, y, z, c = rec

        x0, y0, x1, y1, W, H = _bbox_from_xy(x, y, gsd)

        # DSM: max-z over ALL returns
        dsm, dsm_valid = _max_z_raster(x, y, z, x0, y0, W, H, gsd)

        # DSM_min: min-z over ALL returns. For most cells this equals
        # DSM (single return). For canopy/vegetation cells where LIDAR
        # pierces through gaps, dsm_min often captures ground returns
        # — a strong DTM prior. Used as optional 2nd conditioning
        # channel via cfg.model.unet.in_channel = 3 + cfg.data.use_min_dsm.
        dsm_min, dsm_min_valid = _min_z_raster(x, y, z, x0, y0, W, H, gsd)

        # DTM: TIN over ground+water (cls 2 ∪ 9)
        gmask = np.isin(c, GROUND_CLASSES)
        dtm, dtm_valid = _tin_dtm(x[gmask], y[gmask], z[gmask],
                                   x0, y0, W, H, gsd)

        valid = dsm_valid & dtm_valid
        if not valid.any():
            return scene_id, 0, "no overlapping valid DSM ∩ DTM cells"

        # Joint min-max → [-1, 1] (paper Eq.15) — uses scene-wide DSM
        # and DTM only. dsm_min normalises the same way for consistency.
        dsm_n, dtm_n, stats = joint_minmax_normalise(dsm, dtm, valid)
        v_b = valid.astype(bool)
        s_vmin = float(stats['vmin']); s_vmax = float(stats['vmax'])
        s_span = max(s_vmax - s_vmin, 1e-6)
        dsm_min_n = np.where(v_b,
                             2.0 * (dsm_min - s_vmin) / s_span - 1.0,
                             0.0).astype(np.float32)

        scene_out.mkdir(parents=True, exist_ok=True)

        # Save scene file (for PrioStitch + viz)
        np.savez_compressed(
            scene_marker,
            dsm=dsm.astype(np.float16),
            dsm_min=dsm_min.astype(np.float16),
            dtm=dtm.astype(np.float16),
            valid=valid.astype(np.float32),
            bbox=np.array([x0, y0, x1, y1], dtype=np.float64),
            gsd=np.float32(gsd),
            stats=stats,
        )

        # Slice into tiles. _slice_into_tiles expects (dsm, dtm, valid)
        # triples; we extend to also slice dsm_min by reusing the same
        # tile geometry via a helper.
        tiles = _slice_into_tiles_with_min(dsm_n, dsm_min_n, dtm_n, valid,
                                            tile=tile, stride=stride,
                                            min_valid_frac=min_valid_frac)
        for i, j, d, dmin, g, v in tiles:
            np.savez_compressed(
                scene_out / f"{scene_id}_{i:04d}_{j:04d}.npz",
                dsm_norm=d.astype(np.float16),
                dsm_min_norm=dmin.astype(np.float16),
                dtm_norm=g.astype(np.float16),
                valid_mask=v,
                stats=stats,
                tile_origin=np.array([i, j], dtype=np.int32),
            )
        return scene_id, len(tiles), None
    except Exception as e:
        return scene_id, 0, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


# ---- CLI ---------------------------------------------------------------

def _setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('preprocess')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                             datefmt='%H:%M:%S')
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / 'preprocess.log'); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--laz_root', required=True,
                   help="Either a directory containing per-split subdirs "
                        "of LAZ files (Training/, Test/), or a flat "
                        "directory of LAZ files (use --flat).")
    p.add_argument('--out_dir', required=True,
                   help="Output directory. Will create {out_dir}/{train,test}/.")
    p.add_argument('--gsd', type=float, default=0.5,
                   help="Grid spacing in metres (paper used 0.5 for SU-* and "
                        "1.0 for some others; DEFRA we default 0.5).")
    p.add_argument('--tile', type=int, default=256)
    p.add_argument('--stride', type=int, default=None,
                   help="Tile stride; defaults to non-overlapping (= tile).")
    p.add_argument('--min_valid_frac', type=float, default=0.05)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=None,
                   help="Process at most N LAZ files per split (for smoke tests).")
    p.add_argument('--splits', nargs='+', default=['Training', 'Test'],
                   help="Which subdirs of laz_root to process. Output names "
                        "are auto-lowercased to {train,test}.")
    p.add_argument('--flat', action='store_true',
                   help="Treat --laz_root as a flat directory of .laz files. "
                        "Auto-split by deterministic hash into ~92%% train, "
                        "~8%% test (matches existing 727:63 ratio).")
    p.add_argument('--scene_prefix', default='EN_',
                   help="Scene-id prefix for output dirs (default: EN_, "
                        "matching the existing england_tiles convention).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    log = _setup_logger(out_dir)
    log.info(f"GSD={args.gsd}m  tile={args.tile}  stride={args.stride}  "
             f"workers={args.workers}  ground_classes={GROUND_CLASSES}")

    laz_root = Path(args.laz_root)

    # Build (split_name, [laz_paths]) mapping
    split_files: dict = {}
    if args.flat:
        # Flat: one LAZ → one scene, deterministic 92%/8% train/test split
        # by hash of the filename (matches existing 727:63 ≈ 92:8 ratio).
        import hashlib
        all_laz = sorted(laz_root.rglob('*.la[sz]'),
                          key=lambda p: p.stat().st_size)
        if args.limit:
            all_laz = all_laz[:args.limit]
        for p in all_laz:
            # hash filename → 0..99; <8 → test, else train.
            h = int(hashlib.sha256(p.name.encode()).hexdigest(), 16) % 100
            split = 'test' if h < 8 else 'train'
            split_files.setdefault(split, []).append(p)
        log.info(f"[flat mode] {len(all_laz)} LAZ files → "
                 f"train={len(split_files.get('train', []))}, "
                 f"test={len(split_files.get('test', []))}")
    else:
        # Per-subdir mode (paper convention)
        for split in args.splits:
            split_dir = laz_root / split
            if not split_dir.exists():
                log.warning(f"missing split dir: {split_dir} — skipping")
                continue
            files = sorted(split_dir.rglob('*.la[sz]'),
                            key=lambda p: p.stat().st_size)
            if args.limit:
                files = files[:args.limit]
            split_files[split.lower()] = files

    for split, all_laz in split_files.items():
        out_split = out_dir / split
        out_split.mkdir(parents=True, exist_ok=True)

        log.info(f"[{split}] {len(all_laz)} LAZ files → {out_split}")
        t0 = time.time()
        n_tiles_total = 0; n_scenes_done = 0; n_errors = 0

        worker_args = [
            (str(p), str(out_split), args.gsd, args.tile, args.stride,
             args.min_valid_frac, args.scene_prefix)
            for p in all_laz
        ]

        if args.workers <= 1:
            results = (_process_scene(a) for a in worker_args)
        else:
            ex = ProcessPoolExecutor(max_workers=args.workers)
            futures = [ex.submit(_process_scene, a) for a in worker_args]
            results = (f.result() for f in as_completed(futures))

        for scene_id, n_tiles, err in results:
            n_scenes_done += 1
            n_tiles_total += n_tiles
            if err:
                n_errors += 1
                log.warning(f"  {scene_id}: {err.splitlines()[0]}")
            if n_scenes_done % 25 == 0:
                dt = time.time() - t0
                log.info(f"  [{n_scenes_done}/{len(all_laz)}] "
                         f"{n_tiles_total} tiles  {dt:.0f}s elapsed")

        if args.workers > 1:
            ex.shutdown()
        log.info(f"[{split}] done: {n_scenes_done} scenes, "
                 f"{n_tiles_total} tiles, {n_errors} errors, "
                 f"{time.time() - t0:.0f}s wall")

    log.info("preprocess complete.")


if __name__ == '__main__':
    main()

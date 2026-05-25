"""Pre-tile preprocessed scenes into individual 256x256 .npz tile files.

This eliminates the scene-load stalls that show up at finer GSDs. With
on-the-fly sampling, each worker had to decompress a multi-GB scene .npz
to extract one ~1 MB tile (with `tiles_per_scene_burst` amortising over
multiple tiles). At DEFRA 0.1 m/px (2.8 GB per scene uncompressed),
np.load decompression takes 5-15s wall time, causing the slow steps
visible in the training log.

After pre-tiling, each tile is a single small file (~300-500 KB
compressed) that loads in 2-6 ms cold and sub-ms warm. The dataloader
reads one file per __getitem__, no scene-level cache management
needed.

Output layout:
    {output_root}/{scene_name}/tile_{iy:05d}_{ix:05d}.npz
    {output_root}/{scene_name}/.done   <- marker, presence = scene complete

Each tile file is schema-compatible with the original scene .npz, but
only the windowed region. Per-tile content:
    dsm_max, dsm_min, dsm_mean   (256,256) float32
    dsm_mask, valid, m_alpha     (256,256) uint8
    gt_dtm                       (256,256) float32
    ix, iy                       int32     scalars (tile position in scene)
    gsd, alpha_metres            float32   scalars (passed through)
    scene_bbox                   (4,)      float64 (passed through)

Empty tiles (no valid LiDAR data anywhere in the window) are skipped.
The `.done` marker is written only after all tiles for a scene have
been written, so partial outputs are auto-recovered on re-run.

Usage:
    python -m stage2_raster.scripts.pretile \\
        --input_root  /root/work/stage2_raster_runs/preprocessed \\
        --output_root /root/work/stage2_raster_runs/pretiled \\
        --tile_size 256 --stride 256 --num_workers 16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def _process_scene(args):
    """Worker function. Returns (scene_name, n_written, n_skipped_empty)."""
    scene_dir, out_root, tile, stride = args
    name = scene_dir.name
    src_npz = scene_dir / 'raster.npz'
    if not src_npz.exists():
        return name, 0, 0, f"missing {src_npz}"

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / '.done'

    if done_marker.exists():
        n_done = sum(1 for _ in out_dir.glob('tile_*.npz'))
        return name, n_done, 0, "already done"

    try:
        with np.load(str(src_npz)) as z:
            dsm_max = z['dsm_max']
            dsm_min = z['dsm_min']
            dsm_mean = z['dsm_mean']
            dsm_mask = z['dsm_mask']
            gt_dtm = z['gt_dtm']
            valid = z['valid']
            m_alpha = z['m_alpha']
            bbox = z['bbox'].astype(np.float64)
            gsd = float(z['gsd'])
            alpha_metres = float(z['alpha_metres'])
    except Exception as e:
        return name, 0, 0, f"load failed: {e}"

    H, W = dsm_max.shape
    n_written = 0
    n_skipped = 0

    for iy in range(0, H - tile + 1, stride):
        for ix in range(0, W - tile + 1, stride):
            t_dsm_mask = dsm_mask[iy:iy + tile, ix:ix + tile]
            t_valid = valid[iy:iy + tile, ix:ix + tile]
            # Skip tiles with no LiDAR data and no GT at all (scene edges
            # past the data extent). No information either way → useless
            # to write. Don't apply arbitrary thresholds beyond that;
            # the loss handles partial-coverage tiles via valid_gt
            # masking.
            if int(t_dsm_mask.sum()) == 0 and int(t_valid.sum()) == 0:
                n_skipped += 1
                continue

            tile_path = out_dir / f'tile_{iy:05d}_{ix:05d}.npz'
            np.savez_compressed(
                str(tile_path),
                dsm_max=dsm_max[iy:iy + tile, ix:ix + tile].astype(np.float32, copy=False),
                dsm_min=dsm_min[iy:iy + tile, ix:ix + tile].astype(np.float32, copy=False),
                dsm_mean=dsm_mean[iy:iy + tile, ix:ix + tile].astype(np.float32, copy=False),
                dsm_mask=t_dsm_mask.astype(np.uint8, copy=False),
                gt_dtm=gt_dtm[iy:iy + tile, ix:ix + tile].astype(np.float32, copy=False),
                valid=t_valid.astype(np.uint8, copy=False),
                m_alpha=m_alpha[iy:iy + tile, ix:ix + tile].astype(np.uint8, copy=False),
                ix=np.int32(ix),
                iy=np.int32(iy),
                gsd=np.float32(gsd),
                alpha_metres=np.float32(alpha_metres),
                scene_bbox=bbox,
            )
            n_written += 1

    done_marker.write_text(f"{n_written} tiles, {n_skipped} skipped\n")
    return name, n_written, n_skipped, "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pre-tile preprocessed scenes.")
    ap.add_argument('--input_root', required=True, type=Path,
                     help="Preprocessed scenes directory (where preprocess.py wrote raster.npz)")
    ap.add_argument('--output_root', required=True, type=Path,
                     help="Where to write pre-tiled files")
    ap.add_argument('--tile_size', type=int, default=256)
    ap.add_argument('--stride', type=int, default=256,
                     help="Grid stride. 256 = non-overlapping (paper). "
                          "Smaller = overlap, more tiles, more disk space.")
    ap.add_argument('--num_workers', type=int, default=16,
                     help="Parallel scene-processors. Each holds one decompressed "
                          "scene (~2.8 GB) in RAM. Tune for your box.")
    ap.add_argument('--scenes', nargs='*', default=None,
                     help="Optional: subset of scene names to process (default: all).")
    args = ap.parse_args(argv)

    if not args.input_root.exists():
        sys.exit(f"input_root does not exist: {args.input_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    scene_dirs = []
    for sd in sorted(args.input_root.iterdir()):
        if not sd.is_dir():
            continue
        if not (sd / 'raster.npz').exists():
            continue
        if not (sd / '.done').exists():
            continue
        if args.scenes and sd.name not in args.scenes:
            continue
        scene_dirs.append(sd)

    if not scene_dirs:
        sys.exit(f"No preprocessed scenes found under {args.input_root}.")

    print(f"# pretile")
    print(f"#   input_root  = {args.input_root}")
    print(f"#   output_root = {args.output_root}")
    print(f"#   scenes      = {len(scene_dirs)}")
    print(f"#   tile_size   = {args.tile_size}")
    print(f"#   stride      = {args.stride}  ({'non-overlapping' if args.stride >= args.tile_size else 'overlap'})")
    print(f"#   workers     = {args.num_workers}")
    print()

    try:
        from tqdm.auto import tqdm
        progress = tqdm(total=len(scene_dirs), desc='scenes', unit='scene',
                         dynamic_ncols=True)
    except ImportError:
        progress = None

    tasks = [(sd, args.output_root, args.tile_size, args.stride)
              for sd in scene_dirs]

    t0 = time.time()
    total_written = 0
    total_skipped = 0
    total_errors = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(_process_scene, t) for t in tasks]
        for fut in as_completed(futures):
            name, written, skipped, msg = fut.result()
            total_written += written
            total_skipped += skipped
            if msg not in ("ok", "already done"):
                total_errors += 1
                print(f"[!] {name}: {msg}", flush=True)
            if progress is not None:
                progress.update(1)
                progress.set_postfix(tiles=total_written, skipped=total_skipped,
                                       errors=total_errors)

    if progress is not None:
        progress.close()

    dt = time.time() - t0
    print()
    print(f"# done in {dt:.1f}s")
    print(f"#   tiles written = {total_written}")
    print(f"#   tiles skipped = {total_skipped}  (empty windows)")
    print(f"#   errors        = {total_errors}")
    if total_written > 0:
        # Estimate disk usage from a sample of files.
        sample = list(args.output_root.rglob('tile_*.npz'))[:200]
        if sample:
            mean_size = sum(p.stat().st_size for p in sample) / len(sample)
            est_total_gb = mean_size * total_written / (1024 ** 3)
            print(f"#   ~{mean_size/1024:.0f} KB per tile, "
                  f"~{est_total_gb:.0f} GB total")


if __name__ == '__main__':
    main()

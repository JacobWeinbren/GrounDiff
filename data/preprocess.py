"""Rasterise ALS LAZ scenes into 4-channel input + 1-channel GT + M_α masks.

For each LAZ file, this writes a single `.npz` cache containing:

    dsm_max   (H, W) float32 - max z per cell (paper's DSM channel "s")
    dsm_min   (H, W) float32 - min z per cell
    dsm_mean  (H, W) float32 - mean z per cell
    dsm_mask  (H, W) uint8   - 1 iff cell had >=1 ALS return
    gt_dtm    (H, W) float32 - GT DTM, TIN-interpolated from cls=2/9 returns
    valid     (H, W) uint8   - 1 iff GT DTM is defined (inside ground-return hull)
    m_alpha   (H, W) uint8   - 1 iff |dsm_max − gt_dtm| < alpha_metres
    bbox      (4,)   float64 - (xmin, ymin, xmax, ymax) in CRS units
    gsd       float32        - cell size in metres (e.g. 0.10)
    alpha_metres float32     - threshold used to build M_α (e.g. 0.20)

Per-tile training samples are extracted at dataset-load time. The raster
itself stays in absolute metres-space; per-tile normalisation to [-1, +1]
happens at dataset load (see data/dataset.py).

Empty cells (no return) are filled in `dsm_max`/`dsm_min`/`dsm_mean` with
the local rolling minimum (a sentinel that won't be confused with real
high returns). `dsm_mask` tells the network which cells are real.

Implementation notes:
  - We DON'T smooth or blur. The preprocessor is a deterministic
    rasterisation. All learning happens in the model.
  - GT DTM is built via TIN (Triangulated Irregular Network) using
    scipy.interpolate.LinearNDInterpolator over cls=2 (ground) and cls=9
    (water) returns. Matches the official GrounDiff `clean rebuild`
    preprocess. `valid` is the convex hull of those returns; cells
    outside the hull get z=0 and mask=0.
  - We respect the LAZ classification field exactly: cls=2, 9 are
    ground for the purpose of the GT DTM. M_α is built from the
    residual `|dsm_max − gt_dtm|`, NOT from cls directly — this matches
    paper Eq. 14 and keeps the training target a function of the
    network's inputs.
  - The script is idempotent: completed scenes write a `.done` marker
    keyed on the schema (gsd, alpha). Re-running with a changed alpha
    or gsd invalidates the cache and re-runs only those scenes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np


SCHEMA_VERSION = 3   # v3: GT DTM via TIN (scipy LinearNDInterpolator)
                      # v2: GT DTM via IDW k=8 p=2 (superseded)
                      # v1: legacy
GROUND_CLASSES = (2, 9)


def _read_laz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a LAZ/LAS file and return (xyz [N,3], cls [N,] uint8).

    Filters out non-finite returns. A small number of DEFRA tiles contain
    NaN or +/-inf coordinates (file corruption or bad GNSS post-processing);
    leaving these in would break bbox computation and subsequent
    rasterisation. We drop them silently and raise if nothing's left.
    """
    try:
        import laspy
    except ImportError:
        raise RuntimeError(
            "laspy is required to read LAZ files. "
            "Install with: pip install laspy[laszip]") from None

    with laspy.open(str(path)) as f:
        las = f.read()
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    cls = np.array(las.classification, dtype=np.uint8)

    if xyz.shape[0] == 0:
        raise ValueError(f"empty point cloud in {path.name}")

    finite = np.isfinite(xyz).all(axis=1)
    if not finite.all():
        xyz = xyz[finite]
        cls = cls[finite]
        if xyz.shape[0] == 0:
            raise ValueError(
                f"all points non-finite in {path.name}")
    return xyz, cls


def _rasterise_dsm(xyz: np.ndarray, bbox: Tuple[float, float, float, float],
                    gsd: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                           np.ndarray]:
    """Build per-cell (max, min, mean, count) DSMs.

    Implementation: bucketise each ALS return into its cell, then use
    `numpy.maximum.at` / `numpy.minimum.at` / `numpy.add.at` to fold
    multiple returns per cell. This is O(N) but slower than a sort-based
    impl for very dense scenes — use the sorted variant if profile says.
    """
    xmin, ymin, xmax, ymax = bbox
    W = int(round((xmax - xmin) / gsd))
    H = int(round((ymax - ymin) / gsd))

    # Compute cell indices. We use floor() and clamp to grid.
    ix = np.floor((xyz[:, 0] - xmin) / gsd).astype(np.int64)
    iy = np.floor((xyz[:, 1] - ymin) / gsd).astype(np.int64)
    # Drop out-of-bounds points (shouldn't happen if bbox is from xyz min/max
    # but stay defensive).
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix = ix[ok]; iy = iy[ok]; z = xyz[ok, 2].astype(np.float32)

    # Convention: row-major (H, W). row index = H-1-iy so that the
    # raster top-left corresponds to (xmin, ymax) — standard "north-up"
    # raster orientation for inspection in QGIS etc. Downstream consumers
    # (visualization, LAZ output) need to know this; we'll mirror back
    # at output time.
    row = (H - 1) - iy

    dsm_max = np.full((H, W), -np.inf, dtype=np.float32)
    dsm_min = np.full((H, W), np.inf, dtype=np.float32)
    sum_z = np.zeros((H, W), dtype=np.float64)   # 64-bit accumulator
    count = np.zeros((H, W), dtype=np.int32)

    np.maximum.at(dsm_max, (row, ix), z)
    np.minimum.at(dsm_min, (row, ix), z)
    np.add.at(sum_z, (row, ix), z)
    np.add.at(count, (row, ix), 1)

    mask = (count > 0)
    # Mean = sum / count. Compute in float64 directly into a real
    # accumulator (the previous version wrote into `dsm_mean.astype(...)`
    # which creates a discarded copy — silent zero-out bug).
    mean64 = np.zeros((H, W), dtype=np.float64)
    np.divide(sum_z, count, out=mean64, where=mask)
    dsm_mean = mean64.astype(np.float32)

    # Fill empty cells with a sentinel that's well-below real returns.
    # We use the global min minus 1m. The mask channel tells the model
    # this is a sentinel.
    if mask.any():
        sentinel = float(dsm_max[mask].min()) - 1.0
    else:
        sentinel = 0.0
    dsm_max = np.where(mask, dsm_max, sentinel)
    dsm_min = np.where(mask, dsm_min, sentinel)
    dsm_mean = np.where(mask, dsm_mean, sentinel)

    return (dsm_max.astype(np.float32),
            dsm_min.astype(np.float32),
            dsm_mean.astype(np.float32),
            mask.astype(np.uint8))


def _build_gt_dtm(xyz: np.ndarray, cls: np.ndarray,
                   bbox: Tuple[float, float, float, float],
                   gsd: float, *,
                   k_neighbors: int = 8,
                   idw_power: float = 2.0,
                   max_nn_distance_m: float | None = None,
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Build GT DTM via TIN (Triangulated Irregular Network) interpolation
    of cls={2,9} returns — matches the official GrounDiff `clean rebuild`
    preprocess (`scripts/preprocess.py::_tin_dtm`), which uses
    `scipy.interpolate.LinearNDInterpolator` over the classified ground
    + water returns.

    Why TIN rather than IDW:
      - Official reference uses TIN; matching its data distribution is
        the cleanest paper-faithful prep.
      - The paper's L_∇ magnitude-only loss (Eq. 13) was *specifically
        designed* to be robust to TIN's "arbitrary orientation patterns
        beneath non-ground structures" (paper §3.2). Switching to IDW
        gives a smoother gradient field that the loss wasn't tuned for.

    Validity: cells INSIDE the convex hull of the ground points get
    a valid interpolated value (mask = 1). Cells outside the hull
    (scene corners past every ground return, large no-return regions)
    are NaN from the interpolator → set to 0 with mask=0.

    `k_neighbors`, `idw_power`, `max_nn_distance_m` arguments retained
    on the signature for backward compatibility with any caller still
    passing them (they're ignored by the TIN path).

    Returns (gt_dtm [H,W] float32, valid [H,W] uint8).
    """
    # Args retained for back-compat:
    _ = k_neighbors; _ = idw_power; _ = max_nn_distance_m

    try:
        from scipy.interpolate import LinearNDInterpolator
    except ImportError as e:
        raise SystemExit(
            "scipy is required for TIN DTM. Install with: pip install scipy"
        ) from e

    xmin, ymin, xmax, ymax = bbox
    W = int(round((xmax - xmin) / gsd))
    H = int(round((ymax - ymin) / gsd))

    ground = np.isin(cls, GROUND_CLASSES)
    n_ground = int(ground.sum())
    if n_ground < 3:
        # LinearNDInterpolator needs at least 3 non-colinear points.
        return (np.zeros((H, W), dtype=np.float32),
                np.zeros((H, W), dtype=np.uint8))

    # Subsample very dense ground sets — Delaunay triangulation scales
    # poorly above ~10M points. 4M is plenty for any DEFRA tile.
    MAX_PTS = 4_000_000
    g_xy = xyz[ground, :2].astype(np.float64)
    g_z = xyz[ground, 2].astype(np.float64)
    if g_xy.shape[0] > MAX_PTS:
        rng = np.random.default_rng(0)
        idx = rng.choice(g_xy.shape[0], MAX_PTS, replace=False)
        g_xy = g_xy[idx]; g_z = g_z[idx]

    # Output grid: cell-centre coordinates. The Y axis is image-row
    # convention (row 0 = ymax, row H-1 = ymin) to match dsm_max
    # rasterisation; we sample yy in DESCENDING order so meshgrid maps
    # row 0 to the top of the scene.
    xs = xmin + (np.arange(W, dtype=np.float64) + 0.5) * gsd
    ys = ymax - (np.arange(H, dtype=np.float64) + 0.5) * gsd
    XX, YY = np.meshgrid(xs, ys)

    interp = LinearNDInterpolator(g_xy, g_z, fill_value=np.nan)
    Z = interp(XX, YY).astype(np.float32)
    valid = np.isfinite(Z)
    Z = np.where(valid, Z, 0.0).astype(np.float32)
    return Z, valid.astype(np.uint8)


def _build_m_alpha(dsm_max: np.ndarray, gt_dtm: np.ndarray,
                    valid: np.ndarray, alpha_metres: float) -> np.ndarray:
    """M_α (paper Eq. 14): 1 iff |s − g| < α, in METRES.

    Important: this returns 1 ONLY at cells where the threshold residual
    is small AND `valid_gt` is True. It does NOT check `dsm_mask`; at
    cells with no ALS return, `dsm_max` is a sentinel (well-below all
    real returns), so `|sentinel − gt_dtm|` is large, and M_α will be 0.
    That's the "right wrong answer" — but those cells should be excluded
    from the loss entirely via `loss_valid = dsm_mask & valid_gt`, which
    is built at the dataset layer.

    So: this function's output is M_α-conditional-on-having-real-DSM.
    Use the joint mask `dsm_mask & valid_gt` to gate the loss.
    """
    diff = np.abs(dsm_max - gt_dtm)
    ma = (diff < alpha_metres) & valid.astype(bool)
    return ma.astype(np.uint8)


def _rasterise_ground_returns(xyz: np.ndarray, cls: np.ndarray,
                                bbox: Tuple[float, float, float, float],
                                gsd: float) -> np.ndarray:
    """Build a binary mask: 1 iff the cell has at least one ALS return
    classified as ground (cls=2 or 9). Used by the error visualisation
    to mask out cells where GT_DTM is purely interpolated (i.e. no
    direct measurement of the ground surface inside this cell).
    """
    xmin, ymin, xmax, ymax = bbox
    W = int(round((xmax - xmin) / gsd))
    H = int(round((ymax - ymin) / gsd))
    ground = np.isin(cls, GROUND_CLASSES)
    if not ground.any():
        return np.zeros((H, W), dtype=np.uint8)
    gxy = xyz[ground]
    ix = np.floor((gxy[:, 0] - xmin) / gsd).astype(np.int64)
    iy = np.floor((gxy[:, 1] - ymin) / gsd).astype(np.int64)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix = ix[ok]; iy = iy[ok]
    row = (H - 1) - iy
    out = np.zeros((H, W), dtype=np.uint8)
    out[row, ix] = 1
    return out


def process_one(laz_path: Path, out_dir: Path,
                 gsd: float = 0.10, alpha_metres: float = 0.20,
                 skip_if_done: bool = True, verbose: bool = True
                 ) -> dict | None:
    """Rasterise a single LAZ scene.

    Writes `<out_dir>/<scene_name>/raster.npz` and a `.done` marker.
    Returns a small stats dict or None if skipped.
    """
    scene_name = laz_path.stem.replace('.copc', '')
    scene_out = out_dir / scene_name
    scene_out.mkdir(parents=True, exist_ok=True)
    done_marker = scene_out / '.done'
    npz_path = scene_out / 'raster.npz'

    schema_key = (f"v={SCHEMA_VERSION},gsd={gsd},"
                  f"alpha={alpha_metres}")
    if skip_if_done and done_marker.exists() and npz_path.exists():
        if done_marker.read_text().strip() == schema_key:
            if verbose:
                print(f"  [skip] {scene_name} (cache valid)")
            return None
        # Stale cache — wipe and rebuild.
        if verbose:
            print(f"  [stale] {scene_name} (schema changed) — rebuilding")

    t0 = time.time()
    xyz, cls = _read_laz(laz_path)
    t_read = time.time() - t0

    # Bbox from data, snapped down/up to grid cells.
    xmin = np.floor(xyz[:, 0].min() / gsd) * gsd
    ymin = np.floor(xyz[:, 1].min() / gsd) * gsd
    xmax = np.ceil(xyz[:, 0].max() / gsd) * gsd
    ymax = np.ceil(xyz[:, 1].max() / gsd) * gsd
    bbox = (xmin, ymin, xmax, ymax)

    t0 = time.time()
    dsm_max, dsm_min, dsm_mean, mask = _rasterise_dsm(xyz, bbox, gsd)
    t_dsm = time.time() - t0

    t0 = time.time()
    gt_dtm, valid = _build_gt_dtm(xyz, cls, bbox, gsd)
    t_dtm = time.time() - t0

    had_ground = _rasterise_ground_returns(xyz, cls, bbox, gsd)
    # xyz/cls are no longer needed — free them before the npz write so peak
    # memory drops by ~100-200 MB on big scenes.
    import gc
    n_pts = int(xyz.shape[0])
    n_ground = int(np.isin(cls, GROUND_CLASSES).sum())
    del xyz, cls
    gc.collect()

    m_alpha = _build_m_alpha(dsm_max, gt_dtm, valid, alpha_metres)

    H, W = dsm_max.shape
    n_valid = int(valid.sum())
    pct_ma = 100.0 * float(m_alpha.sum()) / max(int(valid.sum()), 1)
    pct_had_g = 100.0 * float(had_ground.sum()) / (H * W)

    np.savez_compressed(
        npz_path,
        dsm_max=dsm_max, dsm_min=dsm_min, dsm_mean=dsm_mean,
        dsm_mask=mask, gt_dtm=gt_dtm, valid=valid, m_alpha=m_alpha,
        had_ground_return=had_ground,
        bbox=np.array(bbox, dtype=np.float64),
        gsd=np.float32(gsd),
        alpha_metres=np.float32(alpha_metres),
    )
    done_marker.write_text(schema_key)

    if verbose:
        print(f"  [ok]   {scene_name}  "
              f"{H}x{W}px ({W*gsd:.0f}m x {H*gsd:.0f}m)  "
              f"n={n_pts/1e6:.1f}M  ground={n_ground/1e6:.2f}M  "
              f"valid={n_valid/(H*W)*100:.1f}%  M_α={pct_ma:.1f}%  "
              f"had_g={pct_had_g:.1f}%  "
              f"[read={t_read:.1f}s dsm={t_dsm:.1f}s dtm={t_dtm:.1f}s]")

    return dict(
        scene=scene_name, H=H, W=W, n_pts=n_pts, n_ground=n_ground,
        pct_valid=n_valid / (H * W), pct_m_alpha=pct_ma / 100.0,
        bbox=list(bbox), gsd=gsd, alpha_metres=alpha_metres,
        t_total_s=t_read + t_dsm + t_dtm,
    )


def _worker(task):
    """Picklable wrapper around process_one(). Returns a small tuple that
    the parent can stream out as files complete. Catches all exceptions
    so one bad file never kills the pool.
    """
    laz_path, out_dir, gsd, alpha_metres, skip_if_done = task
    try:
        result = process_one(
            Path(laz_path), Path(out_dir),
            gsd=gsd, alpha_metres=alpha_metres,
            skip_if_done=skip_if_done, verbose=False)
        return ('ok', str(laz_path), result, None)
    except Exception as e:
        # Don't print from worker (interleaves badly with other workers).
        # Return the error string and let the parent log + write marker.
        return ('err', str(laz_path), None,
                 f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(
        description="Rasterise LAZ scenes into 4-channel input + GT + M_α.")
    ap.add_argument('--laz_dir', required=True,
                    help="Directory containing input .laz / .copc.laz files.")
    ap.add_argument('--out_dir', required=True,
                    help="Output directory; one subdir per scene.")
    ap.add_argument('--gsd', type=float, default=0.10,
                    help="Cell size in metres. Default 0.10.")
    ap.add_argument('--alpha_metres', type=float, default=0.20,
                    help="Threshold for M_α in metres. Default 0.20 "
                         "(Sithole-Vosselman 2003 §4.2.1).")
    ap.add_argument('--limit', type=int, default=None,
                    help="Max scenes to process this run (for smoke tests).")
    ap.add_argument('--force', action='store_true',
                    help="Ignore .done markers and reprocess.")
    ap.add_argument('--workers', type=int, default=0,
                    help="Number of parallel worker processes. "
                         "Default 0 = (os.cpu_count()//2, capped at 16). "
                         "Each worker loads one LAZ at a time (~0.5–2 GB "
                         "RAM each), so on a 64 GB box keep this <= 16.")
    args = ap.parse_args()

    laz_dir = Path(args.laz_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a deduplicated, sorted list. `*.copc.laz` files also match
    # `*.laz`, so naive list-concat double-counts every one. Use a set
    # keyed by resolved absolute path.
    laz_files = sorted({
        p.resolve()
        for p in (list(laz_dir.glob('*.copc.laz'))
                   + list(laz_dir.glob('*.laz')))
    })
    if args.limit:
        laz_files = laz_files[:args.limit]

    if not laz_files:
        print(f"No .laz / .copc.laz files found in {laz_dir}")
        return

    import os
    if args.workers > 0:
        n_workers = args.workers
    else:
        # Conservative default. Each worker can use 2-4 GB peak on a full
        # DEFRA tile (decoded LAZ + TIN Delaunay + 100M-cell rasters), so
        # 8 workers ≈ 16-32 GB. If you have a beefier box, bump --workers.
        n_workers = min(8, max(1, (os.cpu_count() or 2) // 4))

    print(f"Found {len(laz_files)} LAZ files in {laz_dir}  (deduplicated)")
    print(f"Output: {out_dir}")
    print(f"  gsd       = {args.gsd} m")
    print(f"  alpha     = {args.alpha_metres} m  (Sithole-Vosselman threshold)")
    print(f"  workers   = {n_workers}")
    print()

    schema_key = f"v={SCHEMA_VERSION},gsd={args.gsd},alpha={args.alpha_metres}"
    tasks = []
    n_cached = 0
    for p in laz_files:
        scene_name = p.stem.replace('.copc', '')
        marker = out_dir / scene_name / '.done'
        npz = out_dir / scene_name / 'raster.npz'
        if (not args.force) and marker.exists() and npz.exists():
            try:
                if marker.read_text().strip() == schema_key:
                    n_cached += 1
                    continue
            except Exception:
                pass
        tasks.append((str(p), str(out_dir), args.gsd,
                       args.alpha_metres, not args.force))

    print(f"  {n_cached} scenes already cached and up to date — skipping")
    print(f"  {len(tasks)} scenes to process")
    print()
    if not tasks:
        print("Nothing to do.")
        return

    from concurrent.futures import (
        ProcessPoolExecutor, as_completed, BrokenExecutor)

    stats_all = []
    n_ok = n_err = 0
    t_start = time.time()
    remaining = list(tasks)
    pool_generation = 0
    total = len(remaining)
    progress_i = 0

    # OUTER LOOP: if the pool itself dies (OOM-kill of a worker invalidates
    # the entire executor, all queued futures fail at once with
    # BrokenProcessPool), we collect the unfinished work, halve the worker
    # count, and try again. This keeps us from losing 1000s of files'
    # worth of work when one tile happens to OOM a worker.
    while remaining:
        pool_generation += 1
        if pool_generation > 1:
            print(f"\n--- restarting pool (gen {pool_generation}) with "
                  f"{len(remaining)} tasks remaining at {n_workers} workers ---\n")
        unfinished = list(remaining)
        remaining = []
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_worker, t): t for t in unfinished}
                pending = set(futures)
                for fut in as_completed(futures):
                    pending.discard(fut)
                    progress_i += 1
                    try:
                        status, name, result, err = fut.result()
                    except Exception as e:
                        # BrokenProcessPool or similar — abandon this gen,
                        # everything that didn't run goes to `remaining`.
                        print(f"[{progress_i:>4}/{total}] FAIL     "
                              f"{Path(futures[fut][0]).name}: pool died "
                              f"({type(e).__name__})")
                        n_err += 1
                        # Walk every still-pending future, cancel/collect.
                        for pf in pending:
                            remaining.append(futures[pf])
                        # Halve workers for the retry — usually a memory issue.
                        n_workers = max(1, n_workers // 2)
                        break

                    short = Path(name).name
                    if status == 'ok':
                        if result is None:
                            print(f"[{progress_i:>4}/{total}] cached   {short}")
                        else:
                            stats_all.append(result)
                            n_ok += 1
                            print(f"[{progress_i:>4}/{total}] ok       {short}  "
                                  f"{result['H']}x{result['W']}  "
                                  f"n={result['n_pts']/1e6:.1f}M  "
                                  f"pct_M_α={result['pct_m_alpha']*100:.1f}%  "
                                  f"({result['t_total_s']:.1f}s)")
                    else:
                        n_err += 1
                        print(f"[{progress_i:>4}/{total}] FAIL     "
                              f"{short}: {err}")
                        scene_name = Path(name).stem.replace('.copc', '')
                        fail_dir = out_dir / scene_name
                        fail_dir.mkdir(parents=True, exist_ok=True)
                        (fail_dir / '.failed').write_text(err or 'unknown')
        except BrokenExecutor as e:
            # Belt-and-braces — if the with-block itself raises on cleanup.
            print(f"  pool broke at gen {pool_generation}: "
                  f"{type(e).__name__}: {e}")
            # Any task not in stats_all and not in .failed dir needs retry.
            done_names = {s['scene'] for s in stats_all}
            for t in unfinished:
                sname = Path(t[0]).stem.replace('.copc', '')
                if sname in done_names:
                    continue
                if (out_dir / sname / '.failed').exists():
                    continue
                remaining.append(t)
            n_workers = max(1, n_workers // 2)

        if remaining and n_workers < 1:
            print("workers exhausted — bailing")
            break

    elapsed = time.time() - t_start
    print()
    print(f"Done in {elapsed/60:.1f} min  "
          f"({n_ok} ok, {n_err} failed, {n_cached} cached)  "
          f"final workers={n_workers}")

    summary_path = out_dir / 'preprocess_summary.json'
    # Merge with any prior summary so cached scenes aren't lost.
    prior = []
    if summary_path.exists():
        try:
            prior = json.loads(summary_path.read_text()).get('scenes', [])
        except Exception:
            prior = []
    seen = {s['scene'] for s in stats_all}
    merged = stats_all + [s for s in prior if s.get('scene') not in seen]
    summary_path.write_text(json.dumps(dict(
        gsd=args.gsd, alpha_metres=args.alpha_metres,
        schema_version=SCHEMA_VERSION,
        n_total=len(merged), n_ok_this_run=n_ok,
        n_failed_this_run=n_err, n_cached=n_cached,
        scenes=merged,
    ), indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == '__main__':
    main()

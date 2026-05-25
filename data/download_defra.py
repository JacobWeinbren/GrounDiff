"""Download DEFRA LIDAR 2022 point-cloud tiles from the Flai-hosted
public S3 bucket — no AWS credentials required.

Bucket
------
    s3://open-lidar-data/data/UK/DEFRA/LIDAR_2022/copc/
See https://registry.opendata.aws/open-lidar-data/ and
https://github.com/flai-ai/open-lidar-data

The bucket allows anonymous list + get within this prefix, region
eu-central-1.

Stratified sampling: floor-and-fill
-----------------------------------
England covers ~30 OS 100 km grid pairs (NU, NX, NY, NZ, SD, SE, SH,
SJ, SK, SM, SN, SO, SP, SR, SS, ST, SU, SV, SW, SX, SY, SZ, TA, TF, TG,
TL, TM, TQ, TR, TV). To get geographically-balanced training data we
do a TWO-STAGE selection:

  1. FLOOR — pick at least `min_per_grid` files from EVERY available
     region (or all of them if the region has fewer).
  2. FILL — pick additional files from the larger regions, distributed
     proportionally to remaining file counts, until we reach
     `target_total`.

So with target_total=2000, min_per_grid=30, and ~30 regions: the floor
is 30 × 30 = 900 files (one for each grid square has at least 30
scenes), and the remaining 1100 are filled proportionally from regions
that have more.

File naming
-----------
DEFRA / Flai conventions vary but the OS grid token is always in the
basename. We use a liberal parser that handles both:

    Format A  SU60ne_P_12345_<dates>.copc.laz   (Flai-renamed, prefix at start)
    Format B  P_12345_SU6570_<dates>.copc.laz   (DEFRA original, embedded)

Either way we extract a token like `SU60ne` or `SU6570` and the 100 km
grid prefix is its first two letters (`SU`).

Output
------
A flat directory of `.copc.laz` files at `--laz_root`, named exactly as
they appear in S3. Re-running with new params SKIPS existing files;
add `--target_total 3000` to grow incrementally.

Usage
-----
    python -u -m pipeline.download_defra \\
        --laz_root /root/data/data/england_raw \\
        --target_total 2000 \\
        --min_per_grid 30 \\
        --workers 16

Listing cache
-------------
The bucket has ~6 K keys; first listing takes ~30s. We cache the result
under `~/.cache/groundiff_pp/defra_s3_listing.tsv` for one week so
subsequent runs are instant. Use `--clear_listing_cache` to refresh.
"""
from __future__ import annotations
import argparse
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

S3_BUCKET   = "open-lidar-data"
S3_PREFIX   = "data/UK/DEFRA/LIDAR_2022/copc/"
S3_REGION   = "eu-central-1"

# OS-grid token: 2 uppercase letters followed by 2-6 digits, with an
# optional quadrant suffix (ne/nw/se/sw). Matches both `SU60ne` and
# `SU6570`. Word-boundary anchored.
_GRID_RE = re.compile(r'\b([A-Z]{2}\d{2,6}(?:[ns][ew])?)\b')


def _make_client():
    """Anonymous boto3 client — no AWS credentials needed for this bucket."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client(
        's3',
        config=Config(signature_version=UNSIGNED,
                       region_name=S3_REGION,
                       max_pool_connections=64,
                       retries={'max_attempts': 5, 'mode': 'standard'}))


def _grid_pair_from_key(key: str) -> str | None:
    """Extract the 2-letter OS 100 km grid pair from a key, or None if
    we can't parse it. Liberal — tries multiple methods."""
    name = Path(key).name

    # Method A — basename starts with two uppercase letters then a digit
    # (Flai-renamed layout: SU60ne_P_12345_*.copc.laz)
    if (len(name) >= 3 and name[0].isupper() and name[1].isupper()
            and name[2].isdigit()):
        return name[:2]

    # Method B — find any OS-grid token anywhere in the basename
    m = _GRID_RE.search(name)
    if m:
        return m.group(1)[:2]

    return None


def list_bucket(client, *, force_refresh: bool = False,
                 save_listing: Path | None = None) -> dict[str, list[str]]:
    """Walk the entire bucket prefix once, group keys by 2-letter OS grid
    pair. Caches the result to disk to avoid re-listing on every run."""
    cache = Path.home() / '.cache' / 'groundiff_pp' / 'defra_s3_listing.tsv'
    if not force_refresh and cache.exists() and \
            (time.time() - cache.stat().st_mtime) < 7 * 86400:
        groups: dict[str, list[str]] = defaultdict(list)
        for line in cache.read_text().splitlines():
            try:
                pre, key = line.split('\t', 1)
            except ValueError:
                continue
            groups[pre].append(key)
        n = sum(len(v) for v in groups.values())
        print(f"[s3] using cached listing ({n:,} keys, "
              f"{len(groups)} grid pairs)  [{cache}]", flush=True)
        if save_listing:
            save_listing.write_text(cache.read_text())
            print(f"[s3] saved copy of listing → {save_listing}",
                  flush=True)
        return groups

    print(f"[s3] listing s3://{S3_BUCKET}/{S3_PREFIX} ...", flush=True)
    paginator = client.get_paginator('list_objects_v2')
    groups = defaultdict(list)
    skipped_no_grid = 0
    n = 0; t0 = time.time()
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('.copc.laz'):
                continue
            n += 1
            pre = _grid_pair_from_key(key)
            if pre is None:
                skipped_no_grid += 1
                continue
            groups[pre].append(key)
            if n % 5000 == 0:
                print(f"[s3] ... {n:,} keys listed", flush=True)
    el = time.time() - t0
    parsed = n - skipped_no_grid
    print(f"[s3] listed {n:,} keys in {el:.0f}s  ·  parsed {parsed:,}  "
          f"·  skipped (unparsable) {skipped_no_grid:,}  "
          f"·  {len(groups)} grid pairs", flush=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open('w') as f:
        for pre, keys in sorted(groups.items()):
            for key in keys:
                f.write(f"{pre}\t{key}\n")
    print(f"[s3] cached listing → {cache}", flush=True)
    if save_listing:
        save_listing.write_text(cache.read_text())
        print(f"[s3] saved copy of listing → {save_listing}", flush=True)
    return groups


def stratified_floor_fill(groups: dict[str, list[str]],
                            target_total: int, min_per_grid: int,
                            seed: int = 42,
                            only_grids: list[str] | None = None
                            ) -> tuple[list[str], dict[str, int]]:
    """Floor-and-fill: at least min_per_grid keys from every region, then
    fill up to target_total proportionally to remaining region size.

    Returns (chosen_keys, per_grid_count_dict).
    """
    rng = random.Random(seed)
    available = sorted(groups.keys())
    if only_grids:
        only_set = {g.upper() for g in only_grids}
        available = [g for g in available if g in only_set]

    if not available:
        return [], {}

    # Shuffle each region's keys deterministically
    shuffled: dict[str, list[str]] = {}
    for g in available:
        ks = list(groups[g])
        rng.shuffle(ks)
        shuffled[g] = ks

    chosen: list[str] = []
    per_grid: dict[str, int] = {g: 0 for g in available}

    # 1. FLOOR — min_per_grid from each region (or all if smaller)
    for g in available:
        floor_n = min(len(shuffled[g]), min_per_grid)
        chosen.extend(shuffled[g][:floor_n])
        per_grid[g] = floor_n
        # consume the floor from the region's remaining pool
        shuffled[g] = shuffled[g][floor_n:]

    # 2. FILL — distribute remaining slots proportionally to remaining sizes
    remaining_total = sum(len(ks) for ks in shuffled.values())
    extra_needed = max(0, target_total - len(chosen))
    if remaining_total > 0 and extra_needed > 0:
        # Compute proportional shares; round, then top up if rounding lost a few
        shares: dict[str, int] = {}
        for g, ks in shuffled.items():
            share = round(extra_needed * len(ks) / remaining_total)
            shares[g] = min(share, len(ks))
        # Adjust if rounding undershot or overshot
        diff = extra_needed - sum(shares.values())
        if diff != 0:
            sorted_grids = sorted(shuffled.keys(),
                                    key=lambda g: -len(shuffled[g]))
            i = 0
            while diff != 0 and i < len(sorted_grids) * 5:
                g = sorted_grids[i % len(sorted_grids)]
                if diff > 0 and shares[g] < len(shuffled[g]):
                    shares[g] += 1; diff -= 1
                elif diff < 0 and shares[g] > 0:
                    shares[g] -= 1; diff += 1
                i += 1
        # Add to chosen
        for g in available:
            n = shares[g]
            if n > 0:
                chosen.extend(shuffled[g][:n])
                per_grid[g] += n

    return chosen, per_grid


def _download_one(client, key: str, out_dir: Path,
                   counters: dict, lock: threading.Lock) -> None:
    dst = out_dir / Path(key).name
    if dst.exists() and dst.stat().st_size > 0:
        with lock:
            counters['skipped'] += 1
        return
    tmp = dst.with_suffix(dst.suffix + '.part')
    try:
        client.download_file(S3_BUCKET, key, str(tmp))
        tmp.rename(dst)
        sz = dst.stat().st_size
        with lock:
            counters['ok']    += 1
            counters['bytes'] += sz
    except Exception as e:
        with lock:
            counters['fail'] += 1
        if tmp.exists():
            try: tmp.unlink()
            except OSError: pass
        print(f"[s3] FAIL {Path(key).name}: {type(e).__name__}: {e}",
              flush=True)


def download_files(keys: list[str], out_dir: Path,
                    workers: int = 16) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = _make_client()
    counters = dict(ok=0, skipped=0, fail=0, bytes=0)
    lock = threading.Lock()
    t0 = time.time()
    last_print = t0

    print(f"[s3] downloading {len(keys):,} files → {out_dir} "
          f"({workers} parallel)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download_one, client, k, out_dir,
                           counters, lock): k for k in keys}
        for n, fut in enumerate(as_completed(futs), 1):
            _ = fut.result()
            now = time.time()
            if now - last_print > 4 or n == len(keys):
                el = now - t0
                rate = counters['bytes'] / max(el, 1) / 1e6
                eta = (len(keys) - n) * (el / max(n, 1))
                print(f"[s3] [{n}/{len(keys)}]  "
                      f"ok={counters['ok']} skip={counters['skipped']} "
                      f"fail={counters['fail']}  "
                      f"{counters['bytes']/1e9:.2f} GB  "
                      f"({rate:.1f} MB/s avg)  "
                      f"ETA {eta/60:.1f} min", flush=True)
                last_print = now

    el = time.time() - t0
    print(f"\n[s3] DONE in {el/60:.1f} min  "
          f"ok={counters['ok']} skip={counters['skipped']} "
          f"fail={counters['fail']}  "
          f"{counters['bytes']/1e9:.2f} GB total  "
          f"avg {counters['bytes']/max(el,1)/1e6:.1f} MB/s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--laz_root', required=True,
                    help='Output directory (flat). e.g. /root/data/england_raw')
    ap.add_argument('--target_total', type=int, default=2000,
                    help='Target total number of scenes (default: 2000)')
    ap.add_argument('--min_per_grid', type=int, default=30,
                    help='Floor — minimum scenes per OS 100 km grid pair '
                         '(default: 30). Regions with fewer files contribute '
                         'all of them.')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--grids', default='',
                    help='Comma-separated grid restriction, e.g. "NX,NY,SD". '
                         'Default empty = all available.')
    ap.add_argument('--dry_run', action='store_true',
                    help='List the chosen sample with per-grid counts; '
                         'do not download.')
    ap.add_argument('--clear_listing_cache', action='store_true',
                    help='Force re-listing of the bucket')
    ap.add_argument('--save_listing', default=None,
                    help='Path to save a TSV copy of the bucket listing')
    args = ap.parse_args()

    only_grids = [g.strip().upper() for g in args.grids.split(',')
                  if g.strip()] or None

    client = _make_client()
    groups = list_bucket(client,
                          force_refresh=args.clear_listing_cache,
                          save_listing=Path(args.save_listing)
                                       if args.save_listing else None)

    if not groups:
        print("[s3] FATAL: bucket listing returned 0 keys. "
              "Either the bucket is empty for this prefix, or anonymous "
              "listing is now blocked. Try --clear_listing_cache first; "
              "if that still fails, run "
              "`aws s3 ls s3://open-lidar-data/data/UK/DEFRA/LIDAR_2022/copc/ "
              "--no-sign-request --region eu-central-1 | head` "
              "to confirm bucket access.", flush=True)
        sys.exit(2)

    chosen, per_grid = stratified_floor_fill(
        groups,
        target_total=args.target_total,
        min_per_grid=args.min_per_grid,
        seed=args.seed,
        only_grids=only_grids)

    print(f"\n[plan] sample of {len(chosen):,} scenes from "
          f"{sum(1 for v in per_grid.values() if v > 0)} grid pairs "
          f"(target {args.target_total:,}, floor {args.min_per_grid}/grid)",
          flush=True)
    for g in sorted(per_grid):
        if per_grid[g] == 0: continue
        avail = len(groups.get(g, []))
        flag = '  [FLOOR ONLY]' if per_grid[g] == args.min_per_grid \
                                    and avail > args.min_per_grid else ''
        flag = '  [ALL FILES]' if per_grid[g] == avail else flag
        print(f"   {g}: {per_grid[g]:>4} / {avail:>4} available{flag}",
              flush=True)
    if args.dry_run:
        print("\n[dry-run] no files downloaded. First 10 keys:", flush=True)
        for k in chosen[:10]:
            print(f"   {k}", flush=True)
        return

    out_dir = Path(args.laz_root)
    download_files(chosen, out_dir, workers=args.workers)
    print(f"\n[s3] now run scripts/02_preprocess_stage1.sh", flush=True)


if __name__ == '__main__':
    main()

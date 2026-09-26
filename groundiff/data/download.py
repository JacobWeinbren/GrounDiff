"""Download EA / DEFRA LIDAR 2022 point-cloud tiles (COPC LAZ, classified).

Source: the public, anonymous-read S3 bucket s3://open-lidar-data (Flai),
prefix data/UK/DEFRA/LIDAR_2022/copc/, accessed over plain HTTPS (no AWS
libraries or credentials). Tiles are 500 m OS grid quadrants named like
`NU0445ne_P_12498_20220119_20220119.copc.laz`.

    # geographically balanced sample: >= 20 tiles per 100 km square, 1000 total
    python -m groundiff.data.download --out data/laz --target 1000 --min-per-grid 20
    # everything in given 1 km squares (with all four quadrants), e.g. for a test area
    python -m groundiff.data.download --out data/laz --squares SU6570 SU6571
    # 3x3 blocks of neighbouring quadrants around sampled tiles (context for buffers/splits)
    python -m groundiff.data.download --out data/laz --target 200 --blocks 3

The listing (~tens of thousands of keys) is cached for a week in
~/.cache/groundiff/defra_listing.tsv.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET_URL = "https://open-lidar-data.s3.eu-central-1.amazonaws.com"
PREFIX = "data/UK/DEFRA/LIDAR_2022/copc/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

from .osgrid import os_origin, parse_tile  # noqa: E402  (re-exported)


def list_keys(cache: Path | None = None, refresh: bool = False, timeout: float = 60) -> list[tuple[str, int]]:
    cache = cache or Path.home() / ".cache" / "groundiff" / "defra_listing.tsv"
    if cache.exists() and not refresh and time.time() - cache.stat().st_mtime < 7 * 86400:
        rows = [line.split("\t") for line in cache.read_text().splitlines() if "\t" in line]
        return [(k, int(s)) for k, s in rows]
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": PREFIX, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET_URL}/?{urllib.parse.urlencode(q)}", timeout=timeout) as r:
            root = ET.fromstring(r.read())
        for c in root.iter(f"{NS}Contents"):
            k = c.find(f"{NS}Key").text
            if k.endswith(".laz"):
                keys.append((k, int(c.find(f"{NS}Size").text)))
        token_el = root.find(f"{NS}NextContinuationToken")
        if token_el is None or root.find(f"{NS}IsTruncated").text != "true":
            break
        token = token_el.text
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(f"{k}\t{s}" for k, s in keys))
    return keys


def floor_fill(keys: list[str], target: int, min_per_grid: int, seed: int = 42) -> list[str]:
    """At least `min_per_grid` tiles from every 100 km square, then fill up to
    `target` in proportion to what each square has left."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for k in keys:
        t = parse_tile(k)
        if t:
            groups[t["grid"]].append(k)
    chosen = []
    for g in sorted(groups):
        rng.shuffle(groups[g])
        take = min(min_per_grid, len(groups[g]))
        chosen += groups[g][:take]
        groups[g] = groups[g][take:]
    rest = sum(len(v) for v in groups.values())
    need = max(0, target - len(chosen))
    if rest and need:
        order = sorted(groups, key=lambda g: -len(groups[g]))
        shares = {g: min(len(groups[g]), int(need * len(groups[g]) / rest)) for g in order}
        left = need - sum(shares.values())
        for g in order:
            if left <= 0:
                break
            if shares[g] < len(groups[g]):
                shares[g] += 1
                left -= 1
        for g in order:
            chosen += groups[g][:shares[g]]
    return chosen


def neighbours(keys: list[str], all_keys: list[str], size: int = 3) -> list[str]:
    """Add the size x size block of same-sized tiles (500 m quadrants or 1 km
    squares) centred on each chosen tile, with every file (survey) found at
    each position."""
    by_pos = defaultdict(list)
    for k in all_keys:
        t = parse_tile(k)
        if t:
            by_pos[(t["origin"], t["size"])].append(k)
    out = set(keys)
    r = size // 2
    for k in keys:
        t = parse_tile(k)
        if not t:
            continue
        (x, y), step = t["origin"], t["size"]
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                out.update(by_pos.get(((x + step * i, y + step * j), step), []))
    return sorted(out)


def download(keys: list[str], out: Path, workers: int = 8, timeout: float = 120) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    stats = {"ok": 0, "skipped": 0, "failed": 0, "bytes": 0}

    def one(key):
        dst = out / Path(key).name
        if dst.exists() and dst.stat().st_size > 0:
            return "skipped", 0
        tmp = dst.with_suffix(dst.suffix + ".part")
        url = f"{BUCKET_URL}/{urllib.parse.quote(key)}"
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                tmp.replace(dst)
                return "ok", dst.stat().st_size
            except Exception as e:           # retry with backoff, then report
                err = e
                time.sleep(2 ** attempt)
        tmp.unlink(missing_ok=True)
        print(f"  FAILED {Path(key).name}: {err!r}", file=sys.stderr)
        return "failed", 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, k) for k in keys]
        for i, f in enumerate(as_completed(futs), 1):
            status, size = f.result()
            stats[status] += 1
            stats["bytes"] += size
            if i % 20 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)} ({stats['bytes'] / 1e9:.2f} GB)", flush=True)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=int, default=0, help="number of tiles to sample (floor-and-fill)")
    ap.add_argument("--min-per-grid", type=int, default=20)
    ap.add_argument("--squares", nargs="*", default=[], help="1 km squares to fetch in full, e.g. SU6570")
    ap.add_argument("--grids", nargs="*", default=[], help="restrict to 100 km squares, e.g. SU TQ")
    ap.add_argument("--blocks", type=int, default=1, help="also fetch the NxN block of neighbours (odd N)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refresh-listing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    allk = list_keys(refresh=a.refresh_listing)
    keys = [k for k, _ in allk]
    if a.grids:
        g = {x.upper() for x in a.grids}
        keys = [k for k in keys if (parse_tile(k) or {}).get("grid") in g]
    chosen = []
    if a.squares:
        sq = {s.upper() for s in a.squares}
        chosen += [k for k in keys if (parse_tile(k) or {}).get("square") in sq]
    if a.target:
        chosen += floor_fill(keys, a.target, a.min_per_grid, a.seed)
    if a.blocks > 1:
        chosen = neighbours(chosen, keys, a.blocks)
    chosen = sorted(set(chosen))
    sizes = dict(allk)
    print(f"{len(chosen)} tiles, {sum(sizes.get(k, 0) for k in chosen) / 1e9:.2f} GB")
    if a.dry_run:
        print("\n".join(Path(k).name for k in chosen[:50]))
        return 0
    s = download(chosen, a.out, a.workers)
    print(s)
    return 1 if s["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

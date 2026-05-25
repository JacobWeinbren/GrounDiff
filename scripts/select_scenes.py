#!/usr/bin/env python3
"""Pick N representative scenes for PrioStitch inference + visualization.

Selection: stratified by terrain complexity (= non-ground fraction
within the scene) AND by OS grid prefix (= geographic diversity).
Scenes large enough for PrioStitch (>= 2× tile_size in each dim) are
binned into 5 complexity strata by non-ground fraction, then N is split
evenly across strata with OS-grid-spread inside each stratum.

This avoids the pathology of a "pure random" sample dominated by
ground-saturated rural scenes (where E_T1 and E_T2 percentages explode
because n_nonground is in the hundreds while n_ground is in the
millions). The stratified set spans urban → suburban → rural →
forest-occluded → near-empty, so the val viz tells you something about
how the model handles every regime, not just the modal one.

Per-scene non-ground fraction is computed from the `m_alpha` raster:

    nonground_frac = 1 - ( (m_alpha & valid & dsm_mask).sum() /
                            (valid & dsm_mask).sum() )

m_alpha is paper Eq. 14 in metres (|s − g| < α_m). Where the residual
is small the cell is plausibly ground; where it's large the cell is
non-ground (building/canopy). Restricting to (valid & dsm_mask)
excludes no-data cells from the denominator.

Usage:
    python -m stage2_raster.scripts.select_scenes \\
        --root /path/to/preprocess_root \\
        --tile-size 1024 \\
        --n 20 \\
        --out representative_scenes.json \\
        [--split-json /path/to/split.json]   # restrict to val scenes
        [--no-stratify]                       # fall back to random
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np


def _peek_shape(npz_path: Path) -> tuple[int, int]:
    """Read just (H, W) from the npz's dsm_max header. ~1 ms per scene
    (vs ~700 ms for a full load on a 5000² array)."""
    with zipfile.ZipFile(str(npz_path), 'r') as zf:
        with zf.open('dsm_max.npy', 'r') as f:
            f.read(6)  # numpy magic
            major, _ = f.read(2)
            hlen = struct.unpack('<H' if major == 1 else '<I',
                                  f.read(2 if major == 1 else 4))[0]
            header = f.read(hlen).decode('latin-1')
    return tuple(int(s) for s in ast.literal_eval(header.strip())['shape'])


def _read_nonground_frac(npz_path: Path) -> float | None:
    """Read m_alpha + valid + dsm_mask from the npz and return the
    per-scene non-ground fraction.

    Cost: ~100-400 ms per scene for the three uint8 arrays. Acceptable
    for picking 20 scenes from ~4000 candidates (~10 min one-off).
    Returns None on read failure.
    """
    try:
        with np.load(str(npz_path)) as z:
            m_alpha = z['m_alpha'].astype(bool)
            valid = z['valid'].astype(bool)
            dsm_mask = z['dsm_mask'].astype(bool)
    except (KeyError, zipfile.BadZipFile, OSError):
        return None
    denom_mask = valid & dsm_mask
    n_denom = int(denom_mask.sum())
    if n_denom < 10_000:
        return None
    n_ground = int((m_alpha & denom_mask).sum())
    return 1.0 - (n_ground / n_denom)


_GRID_RE = re.compile(r'^([A-Z]{2})')


def _os_grid(name: str) -> str:
    """Extract the 2-letter OS 100km grid prefix from the scene name.

    DEFRA scene names start with the grid square + 4-digit tile, e.g.
    'TL3478ne_P_12535_20220127_20220130' → 'TL'. Return '' if the
    pattern doesn't match.
    """
    m = _GRID_RE.match(name)
    return m.group(1) if m else ''


def _stratified_pick(candidates: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified sample over non-ground fraction × OS grid.

    Strategy:
      1. Sort candidates by nonground_frac.
      2. Cut into 5 quantile bins → strata = {0..4}, with low-stratum =
         near-empty rural, high-stratum = dense urban.
      3. Allocate n // 5 picks per stratum (+ remainder distributed
         round-robin starting from the densest, since dense scenes are
         more informative for E_T1).
      4. Within each stratum, prefer scenes from DIFFERENT OS grids
         using a greedy round-robin walk over (grid, scene-name)
         deterministically shuffled by `seed`.
      5. If a stratum is undersized (e.g. no extremely-dense scenes in
         the val set), redistribute its remainder back to neighbouring
         strata.
    """
    cands_with_frac = [c for c in candidates if c.get('nonground_frac') is not None]
    if len(cands_with_frac) < n:
        return candidates[:n]

    cands_with_frac.sort(key=lambda c: c['nonground_frac'])
    n_strata = 5
    bins = np.array_split(cands_with_frac, n_strata)
    for s_idx, b in enumerate(bins):
        for c in b:
            c['stratum'] = s_idx

    # Per-stratum picks: split n across strata. If n=20 and 5 strata,
    # 4 per stratum. Extras go to the higher (denser) strata since
    # urban scenes carry more classification signal.
    base = n // n_strata
    extra = n - base * n_strata
    quota = [base] * n_strata
    for i in range(extra):
        quota[-(1 + i)] += 1

    rng = np.random.default_rng(seed)
    picked: list[dict] = []
    leftover_by_stratum: list[list[dict]] = []
    for s_idx, bucket in enumerate(bins):
        bucket = list(bucket)
        rng.shuffle(bucket)

        # Greedy round-robin over OS grid prefixes within the stratum
        # so we don't pull all 4 picks from the same county.
        by_grid: dict[str, list[dict]] = {}
        for c in bucket:
            by_grid.setdefault(c['grid'], []).append(c)
        grid_order = sorted(by_grid.keys(), key=lambda g: (
            -len(by_grid[g]),   # busier grids first
            hashlib.sha256(f"{seed}|{g}".encode()).digest()[:4],
        ))
        cycle: list[dict] = []
        max_per_grid = max(len(v) for v in by_grid.values())
        for layer in range(max_per_grid):
            for g in grid_order:
                if layer < len(by_grid[g]):
                    cycle.append(by_grid[g][layer])
        stratum_picks = cycle[:quota[s_idx]]
        picked.extend(stratum_picks)
        leftover_by_stratum.append(cycle[quota[s_idx]:])

    # If a stratum was undersized (rare; happens only if val has few
    # very-dense or very-sparse scenes), fill from neighbouring strata's
    # leftovers, preferring the next-densest stratum.
    if len(picked) < n:
        # Build a flat ranked leftover list, prioritising mid-strata
        # leftovers (most likely to be informative).
        priority = sorted(range(n_strata),
                           key=lambda s: abs(s - 2))   # 2, 1, 3, 0, 4
        remaining = n - len(picked)
        for s_idx in priority:
            for c in leftover_by_stratum[s_idx]:
                if remaining == 0:
                    break
                picked.append(c)
                remaining -= 1
            if remaining == 0:
                break

    # Final deterministic order: by stratum then by name (so the same
    # selection appears in the same display order across runs).
    picked.sort(key=lambda c: (c.get('stratum', 0), c['name']))
    return picked[:n]


def select_scenes(root: Path, *, tile_size: int = 1024,
                   n: int = 20, min_factor: float = 2.0,
                   restrict_to: set[str] | None = None,
                   seed: int = 0,
                   stratify: bool = True) -> list[dict]:
    """Return up to `n` scenes meeting the size requirement.

    Args:
        root:        preprocess directory (each subdir = scene).
        tile_size:   network input dim (need scenes >= min_factor *
                     tile_size in each dim).
        n:           number of scenes to pick.
        min_factor:  minimum scene side as a multiple of tile_size.
        restrict_to: if provided, only scenes whose name is in the set
                     are considered (e.g. restrict picks to val).
        seed:        controls deterministic shuffles.
        stratify:    if True (default), stratify by non-ground fraction
                     and OS-grid diversity. If False, use the legacy
                     hash-shuffled random pick.

    Each entry: {name, H, W, area_px, nonground_frac, grid, stratum}.
    """
    candidates: list[dict] = []
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    try:
        from tqdm.auto import tqdm as _tqdm
        dir_iter = _tqdm(dirs, desc="picking scenes", unit="scene",
                          leave=False, dynamic_ncols=True)
    except ImportError:
        dir_iter = dirs

    min_side = int(tile_size * min_factor)
    for d in dir_iter:
        if restrict_to is not None and d.name not in restrict_to:
            continue
        npz = d / 'raster.npz'
        if not npz.exists() or not (d / '.done').exists():
            continue
        try:
            H, W = _peek_shape(npz)
        except Exception as e:
            print(f"  [skip] {d.name}: {e}")
            continue
        if min(H, W) < min_side:
            continue
        candidates.append(dict(
            name=d.name,
            H=int(H), W=int(W),
            area_px=int(H * W),
            grid=_os_grid(d.name),
        ))

    if not candidates:
        raise RuntimeError(
            f"No scenes under {root} meet the size criteria "
            f"(need >= {min_side} px in each dim to fit PrioStitch).")

    if not stratify:
        # Legacy: deterministic hash-shuffle, take first n.
        def _hash_key(c: dict) -> int:
            h = hashlib.sha256(f"{seed}|{c['name']}".encode()).digest()
            return int.from_bytes(h[:8], 'little')
        candidates.sort(key=_hash_key)
        return candidates[:n]

    # Compute non-ground fraction for each candidate. Costs ~100-400ms
    # per scene; for picking 20 from ~4000 it's a 10-min one-off and
    # the result is cached in the JSON output so re-runs are free.
    print(f"# computing non-ground fraction across {len(candidates)} "
          f"candidate scenes ...")
    try:
        from tqdm.auto import tqdm as _tqdm
        cand_iter = _tqdm(candidates, desc="m_alpha scan", unit="scene",
                          leave=False, dynamic_ncols=True)
    except ImportError:
        cand_iter = candidates
    for c in cand_iter:
        npz = root / c['name'] / 'raster.npz'
        c['nonground_frac'] = _read_nonground_frac(npz)

    return _stratified_pick(candidates, n, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, type=str,
                    help="Preprocess root directory.")
    ap.add_argument('--tile-size', type=int, default=1024)
    ap.add_argument('--n', type=int, default=20,
                    help="Number of scenes to pick.")
    ap.add_argument('--min-factor', type=float, default=2.0,
                    help="Minimum scene side as a multiple of tile_size.")
    ap.add_argument('--seed', type=int, default=0,
                    help="Deterministic shuffle seed.")
    ap.add_argument('--out', required=True, type=str,
                    help="Output JSON path.")
    ap.add_argument('--split-json', type=str, default=None,
                    help="If set, restrict picks to the val set of the "
                         "given split.json (produced by train.py).")
    ap.add_argument('--no-stratify', action='store_true',
                    help="Use legacy pure-random selection.")
    args = ap.parse_args()

    restrict_to: set[str] | None = None
    if args.split_json:
        sj = json.loads(Path(args.split_json).read_text())
        restrict_to = set(sj.get('val_scenes', []))
        if not restrict_to:
            raise SystemExit(
                f"split JSON at {args.split_json} has no val_scenes")
        print(f"Restricting to {len(restrict_to)} val scenes from "
              f"{args.split_json}")

    sel = select_scenes(Path(args.root), tile_size=args.tile_size,
                          n=args.n, min_factor=args.min_factor,
                          restrict_to=restrict_to,
                          seed=args.seed,
                          stratify=not args.no_stratify)
    print(f"selected {len(sel)} / {args.n} target scenes:")
    if not args.no_stratify:
        print(f"  {'name':<40} {'grid':>4} {'strat':>5} {'%non-g':>7}  "
              f"{'H':>6} {'W':>6}")
        print(f"  {'-'*40:<40} {'-'*4:>4} {'-'*5:>5} {'-'*7:>7}  "
              f"{'-'*6:>6} {'-'*6:>6}")
        for s in sel:
            ng = s.get('nonground_frac')
            ng_s = f"{100*ng:6.1f}%" if ng is not None else "    n/a"
            print(f"  {s['name']:<40} {s.get('grid',''):>4} "
                  f"{s.get('stratum','?'):>5}  {ng_s:>7}  "
                  f"{s['H']:>6} {s['W']:>6}")
    else:
        print(f"  {'name':<30} {'H':>6} {'W':>6}  area_px")
        print(f"  {'-'*30:<30} {'-'*6:>6} {'-'*6:>6}  -------")
        for s in sel:
            print(f"  {s['name']:<30} {s['H']:>6} {s['W']:>6}  "
                  f"{s['area_px']:>10}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        tile_size=args.tile_size,
        min_factor=args.min_factor,
        seed=args.seed,
        n_target=args.n,
        n_selected=len(sel),
        stratified=not args.no_stratify,
        scenes=sel,
    ), indent=2))
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()

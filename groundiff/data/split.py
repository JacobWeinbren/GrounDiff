"""Spatially blocked train/val/test split of scenes.

Neighbouring tiles share terrain, surveys and editors, so random tile splits
leak. Scenes are grouped into square blocks (default 10 km, by scene centre)
and whole blocks are assigned to splits, in the spirit of ALS2DTM's
quarter-based split and ResDepth's stripe allocation. Scenes on either side
of a block edge can still be neighbours; with 500 m tiles and 10 km blocks
that affects about 10 % of scenes, at the block boundary only.

    python -m groundiff.data.split --root data/scenes --out data/split.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def block_split(centres: dict[str, tuple[float, float]], block_m: float = 10_000.0,
                fractions=(0.8, 0.1, 0.1), seed: int = 0) -> dict[str, list[str]]:
    """Assign whole blocks to train/val/test, greedily by remaining deficit
    (largest block first, to the split furthest below its target share), so
    one big block cannot swallow a small split. Needs at least 3 blocks."""
    blocks = defaultdict(list)
    for name, (cx, cy) in centres.items():
        blocks[(int(np.floor(cx / block_m)), int(np.floor(cy / block_m)))].append(name)
    if len(blocks) < 3:
        raise ValueError(f"only {len(blocks)} spatial block(s) of {block_m / 1000:g} km: need at least 3 for "
                         "train/val/test; use a smaller --block-km or more scattered scenes")
    names = ("train", "val", "test")
    frac = np.asarray(fractions, np.float64) / np.sum(fractions)
    total = sum(len(v) for v in blocks.values())
    target = frac * total
    rng = np.random.default_rng(seed)
    keys = sorted(blocks)
    rng.shuffle(keys)                                        # random tie-breaking
    keys.sort(key=lambda k: -len(blocks[k]))                  # stable: largest blocks first
    got = np.zeros(3)
    out = {n: [] for n in names}
    # every split gets one block first (the smallest ones for val/test)
    for i, n in ((1, "val"), (2, "test"), (0, "train")):
        k = keys.pop() if n != "train" else keys.pop(0)
        out[n] += sorted(blocks[k])
        got[i] += len(blocks[k])
    for k in keys:
        i = int(np.argmax((target - got) / np.maximum(target, 1e-9)))
        out[names[i]] += sorted(blocks[k])
        got[i] += len(blocks[k])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--block-km", type=float, default=10.0)
    ap.add_argument("--fractions", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-suspect", action="store_true", help="also split scenes the quality gate flagged")
    a = ap.parse_args(argv)
    centres = {}
    for m in sorted(a.root.glob("*/meta.json")):
        meta = json.loads(m.read_text())
        if meta.get("quality", {}).get("suspect") and not a.include_suspect:
            continue
        g = meta["grid"]
        cx = g["xmin"] + 0.5 * g["width"] * g["gsd"]
        cy = g["ymax"] - 0.5 * g["height"] * g["gsd"]
        centres[m.parent.name] = (cx, cy)
    try:
        split = block_split(centres, a.block_km * 1000.0, tuple(a.fractions), a.seed)
    except ValueError as e:
        raise SystemExit(str(e))
    split["block_km"] = a.block_km
    a.out.write_text(json.dumps(split, indent=1))
    print({k: len(v) for k, v in split.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()

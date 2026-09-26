"""Spatially blocked train/val/test split of scenes.

Neighbouring tiles share terrain, surveys and editors, so random tile splits
leak. Scenes are grouped into square blocks (default 10 km, by scene centre)
and whole blocks are assigned to splits, in the spirit of ALS2DTM's
quarter-based split and ResDepth's stripe allocation.

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
    blocks = defaultdict(list)
    for name, (cx, cy) in centres.items():
        blocks[(int(cx // block_m), int(cy // block_m))].append(name)
    keys = sorted(blocks)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    total = sum(len(v) for v in blocks.values())
    out = {"train": [], "val": [], "test": []}
    targets = np.cumsum(fractions) / np.sum(fractions) * total
    running = 0
    for key in keys:
        split = "train" if running < targets[0] else ("val" if running < targets[1] else "test")
        out[split] += sorted(blocks[key])
        running += len(blocks[key])
    # guarantee non-empty val/test when there are at least 3 blocks
    for s in ("val", "test"):
        if not out[s] and len(keys) >= 3 and out["train"]:
            donor_key = next(k for k in reversed(keys) if blocks[k][0] in out["train"])
            moved = blocks[donor_key]
            out["train"] = [n for n in out["train"] if n not in moved]
            out[s] += moved
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--block-km", type=float, default=10.0)
    ap.add_argument("--fractions", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    centres = {}
    for m in sorted(a.root.glob("*/meta.json")):
        g = json.loads(m.read_text())["grid"]
        cx = g["xmin"] + 0.5 * g["width"] * g["gsd"]
        cy = g["ymax"] - 0.5 * g["height"] * g["gsd"]
        centres[m.parent.name] = (cx, cy)
    split = block_split(centres, a.block_km * 1000.0, tuple(a.fractions), a.seed)
    split["block_km"] = a.block_km
    a.out.write_text(json.dumps(split, indent=1))
    print({k: len(v) for k, v in split.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()

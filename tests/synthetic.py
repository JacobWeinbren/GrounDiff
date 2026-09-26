"""Synthetic ALS scenes for tests: terrain + embankment + building + trees + noise.

`write_scene` writes an "after" LAS (hand-edited classes: 2 ground, 5
vegetation, 6 building, 7/18 noise) and a "before" LAS with the same points
classified like lasground_new output (1/2 only) where two typical
lasground_new mistakes are planted: the building roof is kept as ground, and
the embankment crest is dropped from ground.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def terrain(x, y):
    return 50.0 + 0.02 * x + 0.01 * y + 1.5 * np.exp(-((x - 60) ** 2) / 30.0)   # embankment ridge at x=60


def make_points(size: float = 128.0, density: float = 8.0, seed: int = 0, x0: float = 400000.0,
                y0: float = 200000.0):
    rng = np.random.default_rng(seed)
    n = int(size * size * density)
    lx, ly = rng.uniform(0, size, n), rng.uniform(0, size, n)
    z = terrain(lx, ly) + rng.normal(0, 0.02, n)
    cls = np.full(n, 2, np.uint8)
    rn = np.ones(n, np.uint8)
    nr = np.ones(n, np.uint8)
    # building: 20x20 m box, roof 8 m above ground, no ground returns underneath
    b = (lx > 20) & (lx < 40) & (ly > 20) & (ly < 40)
    z[b] = terrain(lx[b], ly[b]) + 8.0
    cls[b] = 6
    # trees: canopy first returns at +12 m, last returns on the ground beneath
    t = (lx > 80) & (lx < 110) & (ly > 70) & (ly < 110)
    tree_idx = np.flatnonzero(t)
    canopy = tree_idx[: tree_idx.size // 2]
    z[canopy] = terrain(lx[canopy], ly[canopy]) + 12.0 + rng.normal(0, 0.5, canopy.size)
    cls[canopy] = 5
    rn[canopy], nr[canopy] = 1, 2
    under = tree_idx[tree_idx.size // 2:]
    rn[under], nr[under] = 2, 2
    # noise
    k = 20
    nx, ny = rng.uniform(0, size, 2 * k), rng.uniform(0, size, 2 * k)
    nz = terrain(nx, ny) + np.r_[np.full(k, -15.0), np.full(k, 60.0)]
    ncls = np.r_[np.full(k, 7, np.uint8), np.full(k, 18, np.uint8)]
    lx, ly, z = np.r_[lx, nx], np.r_[ly, ny], np.r_[z, nz]
    cls = np.r_[cls, ncls]
    rn, nr = np.r_[rn, np.ones(2 * k, np.uint8)], np.r_[nr, np.ones(2 * k, np.uint8)]
    return lx + x0, ly + y0, z, cls, rn, nr


def write_las(path: Path, x, y, z, cls, rn, nr):
    import laspy

    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.offsets = [np.floor(x.min()), np.floor(y.min()), np.floor(z.min())]
    hdr.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = x, y, z
    las.classification = cls
    las.return_number = rn
    las.number_of_returns = nr
    las.write(str(path))


def write_scene(folder: Path, name: str = "SX0000_test", seed: int = 0, size: float = 128.0):
    folder = Path(folder)
    (folder / "after").mkdir(parents=True, exist_ok=True)
    (folder / "before").mkdir(parents=True, exist_ok=True)
    x, y, z, cls, rn, nr = make_points(size=size, seed=seed)
    write_las(folder / "after" / f"{name}.las", x, y, z, cls, rn, nr)
    lx, ly = x - x.min(), y - y.min()
    before = np.where(np.isin(cls, (2,)), 2, 1).astype(np.uint8)   # lasground_new writes only 1/2
    roof = cls == 6
    before[roof] = 2                                              # roof kept as ground
    crest = (cls == 2) & (np.abs(lx - 60) < 2)
    before[crest] = 1                                             # crest cut off
    write_las(folder / "before" / f"{name}.las", x, y, z, before, rn, nr)
    return folder / "after" / f"{name}.las", folder / "before" / f"{name}.las"

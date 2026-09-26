"""Point cloud -> per-cell rasters.

Grid convention (all rasters): north-up, row 0 is the northern edge.
Cell (row, col) covers x in [xmin + col*gsd, xmin + (col+1)*gsd) and
y in (ymax - (row+1)*gsd, ymax - row*gsd]. TIN surfaces are evaluated at
cell centres, so DSM statistics and DTMs refer to the same cells.

Channels produced (all float32, metres unless stated):
  dsm_max    highest return per cell (GrounDiff's DSM: max rasterisation,
             supplement §7.1; ALS2DTM calls this voxel-top)
  dsm_min    lowest return per cell (ALS2DTM voxel-bottom analogue)
  dsm_last   lowest last-or-only return per cell (EA-style last-return DSM)
  density    returns per m²                         (ALS2DTM statistic raster)
  z_std      std of return heights in the cell      (ALS2DTM statistic raster)
  echoes     mean number_of_returns in the cell     (ALS2DTM statistic raster)
  has_return 1 where the cell holds at least one return
A TIN DTM is built from the points of chosen classes (see `tin_dtm`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    xmin: float
    ymax: float
    gsd: float
    width: int
    height: int

    @classmethod
    def from_bounds(cls, xmin, ymin, xmax, ymax, gsd):
        x0 = np.floor(xmin / gsd) * gsd
        y1 = np.ceil(ymax / gsd) * gsd
        w = int(np.ceil((xmax - x0) / gsd - 1e-9))
        h = int(np.ceil((y1 - ymin) / gsd - 1e-9))
        return cls(float(x0), float(y1), float(gsd), max(w, 1), max(h, 1))

    @property
    def bounds(self):
        return (self.xmin, self.ymax - self.height * self.gsd,
                self.xmin + self.width * self.gsd, self.ymax)

    def cell_index(self, x, y):
        col = np.floor((x - self.xmin) / self.gsd).astype(np.int64)
        row = np.floor((self.ymax - y) / self.gsd).astype(np.int64)
        return row, col

    def cell_centres(self):
        xs = self.xmin + (np.arange(self.width) + 0.5) * self.gsd
        ys = self.ymax - (np.arange(self.height) + 0.5) * self.gsd
        return xs, ys

    def to_dict(self):
        return {"xmin": self.xmin, "ymax": self.ymax, "gsd": self.gsd,
                "width": self.width, "height": self.height}


def _flat_index(grid: Grid, x, y):
    row, col = grid.cell_index(x, y)
    ok = (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)
    return row * grid.width + col, ok


def rasterise_points(grid: Grid, x, y, z, return_number=None, number_of_returns=None) -> dict:
    n_cells = grid.width * grid.height
    idx, ok = _flat_index(grid, x, y)
    idx, z = idx[ok], z[ok].astype(np.float64)
    count = np.bincount(idx, minlength=n_cells).astype(np.float64)
    has = count > 0

    zmax = np.full(n_cells, -np.inf)
    zmin = np.full(n_cells, np.inf)
    np.maximum.at(zmax, idx, z)
    np.minimum.at(zmin, idx, z)
    s1 = np.bincount(idx, weights=z, minlength=n_cells)
    mean = np.where(has, s1 / np.maximum(count, 1), 0.0)
    # two-pass variance for numerical stability at 100+ m elevations
    dev = z - mean[idx]
    var = np.bincount(idx, weights=dev * dev, minlength=n_cells) / np.maximum(count, 1)

    out = {
        "dsm_max": np.where(has, zmax, np.nan),
        "dsm_min": np.where(has, zmin, np.nan),
        "density": count / (grid.gsd * grid.gsd),
        "z_std": np.where(has, np.sqrt(var), 0.0),
        "has_return": has.astype(np.float64),
    }
    if return_number is not None and number_of_returns is not None:
        rn, nr = return_number[ok], number_of_returns[ok]
        last = rn >= nr                                   # last or only return
        zlast = np.full(n_cells, np.inf)
        np.minimum.at(zlast, idx[last], z[last])
        out["dsm_last"] = np.where(np.isfinite(zlast), zlast, np.nan)
        echo_sum = np.bincount(idx, weights=nr.astype(np.float64), minlength=n_cells)
        out["echoes"] = np.where(has, echo_sum / np.maximum(count, 1), 0.0)
    return {k: v.reshape(grid.height, grid.width).astype(np.float32) for k, v in out.items()}


def class_mode_onehot(grid: Grid, x, y, is_ground) -> np.ndarray:
    """DeepTerRa's semantic raster: the mode label of each cell's points,
    binarised to ground / non-ground and one-hot encoded as 2 channels
    (ALS2DTM §V-E, "sem2"). Empty cells are [0, 0]; ties go to ground."""
    n_cells = grid.width * grid.height
    idx, ok = _flat_index(grid, x, y)
    idx, g = idx[ok], is_ground[ok]
    ng = np.bincount(idx[g], minlength=n_cells)
    nn_ = np.bincount(idx[~g], minlength=n_cells)
    has = (ng + nn_) > 0
    ground = has & (ng >= nn_)
    nonground = has & ~ground
    return np.stack([ground, nonground]).reshape(2, grid.height, grid.width).astype(np.float32)


def top_return_is(grid: Grid, x, y, z, flag) -> np.ndarray:
    """1 where the highest return in the cell has `flag` set, 0 where not,
    NaN for empty cells. Used as an alternative M_alpha target ("the DSM is
    ground here"), which avoids the |s - g| < alpha rule labelling steep
    ground as non-ground."""
    n_cells = grid.width * grid.height
    idx, ok = _flat_index(grid, x, y)
    idx, z, flag = idx[ok], z[ok], flag[ok]
    order = np.lexsort((z, idx))
    last = np.ones(order.size, bool)
    last[:-1] = idx[order][1:] != idx[order][:-1]
    top = order[last]
    out = np.full(n_cells, np.nan)
    out[idx[top]] = flag[top].astype(np.float64)
    return out.reshape(grid.height, grid.width).astype(np.float32)


def tin_dtm(grid: Grid, x, y, z, max_points: int = 4_000_000, seed: int = 0):
    """Linear interpolation on a Delaunay triangulation (TIN) of the given
    points, evaluated at cell centres. Returns (dtm, valid) where valid is
    False outside the convex hull. Very dense inputs are thinned by a
    per-cell minimum first (keeps the lowest point in each cell), then
    randomly if still above `max_points`."""
    from scipy.interpolate import LinearNDInterpolator

    if x.size < 3:
        return (np.full((grid.height, grid.width), np.nan, np.float32),
                np.zeros((grid.height, grid.width), bool))
    if x.size > max_points:
        # keep the lowest point per cell, which preserves the terrain surface
        idx, ok = _flat_index(grid, x, y)
        order = np.lexsort((z[ok], idx[ok]))
        first = np.ones(order.size, bool)
        first[1:] = idx[ok][order][1:] != idx[ok][order][:-1]
        keep = np.flatnonzero(ok)[order[first]]
        x, y, z = x[keep], y[keep], z[keep]
        if x.size > max_points:
            sel = np.random.default_rng(seed).choice(x.size, max_points, replace=False)
            x, y, z = x[sel], y[sel], z[sel]
    xs, ys = grid.cell_centres()
    # interpolate in local coordinates: Delaunay on 6-7 digit BNG values loses precision
    ox, oy = grid.xmin, grid.ymax
    interp = LinearNDInterpolator(np.column_stack([x - ox, y - oy]), z.astype(np.float64),
                                  fill_value=np.nan)
    XX, YY = np.meshgrid(xs - ox, ys - oy)
    dtm = interp(XX, YY).astype(np.float32)
    return dtm, np.isfinite(dtm)


def build_rasters(grid: Grid, pts, before=None, ground_classes=(2, 9), before_ground_classes=(2,),
                  with_target: bool = True) -> dict:
    """Every channel used in training/inference for one grid. `pts` is the
    final ("after") point set, `before` the lasground_new-classified one."""
    out = rasterise_points(grid, pts.x, pts.y, pts.z, pts.return_number, pts.number_of_returns)
    if with_target:
        g = np.isin(pts.cls, np.asarray(ground_classes, np.uint8))
        gt, gt_valid = tin_dtm(grid, pts.x[g], pts.y[g], pts.z[g])
        out["gt_dtm"], out["gt_valid"] = gt, gt_valid.astype(np.float32)
        out["top_ground"] = top_return_is(grid, pts.x, pts.y, pts.z, g)
    if before is not None:
        bg = np.isin(before.cls, np.asarray(before_ground_classes, np.uint8))
        dtm_b, b_valid = tin_dtm(grid, before.x[bg], before.y[bg], before.z[bg])
        out["dtm_before"], out["before_valid"] = dtm_b, b_valid.astype(np.float32)
        sem = class_mode_onehot(grid, before.x, before.y, bg)
        out["sem_ground"], out["sem_nonground"] = sem[0], sem[1]
    return out

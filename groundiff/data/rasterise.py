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
  in_survey  1 inside the LiDAR coverage (returns plus voids narrower than 60 m
             such as water; see normalise.coverage_mask)
A TIN DTM is built from the points of chosen classes (see `tin_dtm`) and is
cut to in_survey, so no triangle spanning open sea or the survey edge
counts as a target or input. Targets can instead come from published DTM
rasters (`target_from_rasters`).
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


def tin_dtm(grid: Grid, x, y, z):
    """Linear interpolation on a Delaunay triangulation (TIN) of all given
    points, evaluated at cell centres. Returns (dtm, valid) where valid is
    False outside the convex hull. No thinning, so training scenes and
    inference blocks of any size give the same surface (memory is roughly
    300 bytes per point: keep inference blocks to ~1 km)."""
    from scipy.interpolate import LinearNDInterpolator

    if x.size < 3:
        return (np.full((grid.height, grid.width), np.nan, np.float32),
                np.zeros((grid.height, grid.width), bool))
    xs, ys = grid.cell_centres()
    # interpolate in local coordinates: Delaunay on 6-7 digit BNG values loses precision
    ox, oy = grid.xmin, grid.ymax
    interp = LinearNDInterpolator(np.column_stack([x - ox, y - oy]), z.astype(np.float64),
                                  fill_value=np.nan)
    XX, YY = np.meshgrid(xs - ox, ys - oy)
    dtm = interp(XX, YY).astype(np.float32)
    return dtm, np.isfinite(dtm)


def input_rasters(grid: Grid, pts, lasground: bool = True, before_ground_classes=(2,),
                  coverage_close_m: float = 30.0) -> dict:
    """The network's input rasters from one point set: the DSM / statistic
    rasters and in_survey, plus (lasground=True) dtm_before, before_valid and
    sem_* from its classification, which must be lasground_new's (default
    settings write only 1 = non-ground, 2 = ground). The same function serves
    training (preprocess) and inference (batch / QGIS), so both see the same
    points: nothing is dropped by class, because published EA classes are
    not the ones the DTM was made from and production tiles have none yet."""
    from ..normalise import coverage_mask

    out = rasterise_points(grid, pts.x, pts.y, pts.z, pts.return_number, pts.number_of_returns)
    survey = coverage_mask(out["has_return"] > 0, grid.gsd, coverage_close_m)
    out["in_survey"] = survey.astype(np.float32)
    if lasground:
        bg = np.isin(pts.cls, np.asarray(before_ground_classes, np.uint8))
        dtm_b, b_valid = tin_dtm(grid, pts.x[bg], pts.y[bg], pts.z[bg])
        b_valid &= survey
        out["dtm_before"] = np.where(b_valid, dtm_b, np.nan).astype(np.float32)
        out["before_valid"] = b_valid.astype(np.float32)
        sem = class_mode_onehot(grid, pts.x, pts.y, bg)
        out["sem_ground"], out["sem_nonground"] = sem[0], sem[1]
    return out


def target_from_points(grid: Grid, after, survey: np.ndarray, ground_classes=(2,)) -> dict:
    """Target DTM from a hand-edited point cloud (e.g. your own LP360 tiles):
    TIN of the ground class. EA production uses 1 unclassified, 2 ground,
    bridge and low noise, and builds the DTM from ground only."""
    g = np.isin(after.cls, np.asarray(ground_classes, np.uint8))
    gt, gt_valid = tin_dtm(grid, after.x[g], after.y[g], after.z[g])
    gt_valid &= survey
    return {"gt_dtm": np.where(gt_valid, gt, np.nan).astype(np.float32),
            "gt_valid": gt_valid.astype(np.float32),
            "top_ground": top_return_is(grid, after.x, after.y, after.z, g)}


def flat_areas(z: np.ndarray, min_cells: int = 50) -> np.ndarray:
    """Connected areas of exactly equal value (4-neighbours) of at least
    min_cells: hydro-flattened water in EA DTMs (e.g. a river set to 2.48 m
    over 100,000+ cells). Natural terrain gives equal runs of a few cells at
    most, even with values rounded to the millimetre."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    H, W = z.shape
    idx = np.arange(H * W).reshape(H, W)
    fin = np.isfinite(z)
    eh = fin[:, :-1] & fin[:, 1:] & (z[:, :-1] == z[:, 1:])
    ev = fin[:-1, :] & fin[1:, :] & (z[:-1, :] == z[1:, :])
    a = np.concatenate([idx[:, :-1][eh], idx[:-1, :][ev]])
    b = np.concatenate([idx[:, 1:][eh], idx[1:, :][ev]])
    if a.size == 0:
        return np.zeros((H, W), bool)
    g = coo_matrix((np.ones(a.size, np.int8), (a, b)), shape=(H * W, H * W))
    _, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    return (sizes[lab] >= min_cells).reshape(H, W)


def target_from_rasters(grid: Grid, paths: list, survey: np.ndarray, flat_min_cells: int = 50) -> dict:
    """Target DTM from published DTM rasters (the EA product), sampled at
    our cell centres (exact copy when the grids coincide, e.g. gsd 1 m on
    the whole-metre OS grid). Hydro-flattened water (flat_areas) is excluded
    from the target: its level is a production choice the points cannot show."""
    from ..io_raster import sample_rasters

    xs, ys = grid.cell_centres()
    gt = sample_rasters(paths, xs, ys)
    ok = np.isfinite(gt) & survey
    flat = flat_areas(gt, flat_min_cells) if flat_min_cells else np.zeros_like(ok)
    ok &= ~flat
    return {"gt_dtm": np.where(ok, gt, np.nan).astype(np.float32), "gt_valid": ok.astype(np.float32),
            "flat_water": flat.astype(np.float32)}


def build_rasters(grid: Grid, pts, before=None, ground_classes=(2,), before_ground_classes=(2,),
                  with_target: bool = True, coverage_close_m: float = 30.0) -> dict:
    """Inputs from `before` (lasground_new-classified) when given, else from
    `pts` without lasground channels; target (with_target) from the ground
    class of `pts` (a hand-edited point cloud)."""
    src = before if before is not None else pts
    out = input_rasters(grid, src, before is not None, before_ground_classes, coverage_close_m)
    if with_target:
        out.update(target_from_points(grid, pts, out["in_survey"] > 0.5, ground_classes))
    return out

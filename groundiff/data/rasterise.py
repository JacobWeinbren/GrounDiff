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

import os
from concurrent.futures import ThreadPoolExecutor
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


def tin_dtm(grid: Grid, x, y, z, block_points: int = 300_000, workers: int | None = None, need=None):
    """Linear interpolation on a Delaunay triangulation (TIN) of all given
    points, evaluated at cell centres. Returns (dtm, valid) where valid is
    False outside the convex hull (and where need, if given, is False).
    No thinning."""
    # local coordinates: Delaunay on 6-7 digit BNG values loses precision
    return tin_dtm_local(grid, np.asarray(x) - grid.xmin, np.asarray(y) - grid.ymax, z, block_points, workers,
                         need)


def tin_dtm_local(grid: Grid, lx, ly, z, block_points: int = 300_000, workers: int | None = None, need=None):
    """tin_dtm with coordinates relative to (grid.xmin, grid.ymax) (ly <= 0),
    in any float dtype (float32 keeps big tiles small).

    Large inputs are triangulated in blocks of ~block_points in parallel
    threads (Qhull slows down sharply beyond a few 100k points and needs
    ~300 bytes per point). Each block sees the points within a margin around
    it, and a cell's value is only accepted when the circumcircle of its
    triangle lies inside the points the block saw: such a triangle is then a
    triangle of the full Delaunay triangulation (empty-circle property), so
    the result equals one TIN of all points. Other cells (large voids:
    water, buildings) are redone with a wider margin. need: bool [H, W] of
    the cells wanted (e.g. the survey coverage; the rest stay NaN, which also
    spares the huge triangles across open sea or beyond the survey edge)."""
    H, W, gsd = grid.height, grid.width, grid.gsd
    dtm = np.full((H, W), np.nan, np.float32)
    n = int(np.size(lx))
    if n < 3:
        return dtm, np.zeros((H, W), bool)
    lx, ly = np.asarray(lx), np.asarray(ly)
    z = np.asarray(z)
    ext = (float(lx.min()), float(ly.min()), float(lx.max()), float(ly.max()))     # all points
    need = np.ones((H, W), bool) if need is None else np.asarray(need, bool)

    def wanted(r0, r1, c0, c1):
        rr, cc = np.nonzero(need[r0:r1, c0:c1])
        return rr + r0, cc + c0

    if n <= 2 * block_points:
        cells = wanted(0, H, 0, W)
        if cells[0].size:
            _tin_cells(dtm, grid, lx, ly, z, (0, H, 0, W), cells, ext)
        return dtm, np.isfinite(dtm)
    density = n / max(W * H * gsd * gsd, 1e-9)
    side = int(np.clip(np.ceil(np.sqrt(block_points / density) / gsd), 32, max(H, W)))
    margin = max(15.0, 0.1 * side * gsd)
    workers = workers or min(8, os.cpu_count() or 1)
    bins = _Bins(lx, ly, side * gsd, int(np.ceil(W / side)), int(np.ceil(H / side)))

    def gather(reg):
        i = bins.query(reg)
        return lx[i], ly[i], z[i]

    jobs = []
    for c0 in range(0, W, side):
        c1 = min(c0 + side, W)
        for r0 in range(0, H, side):
            r1 = min(r0 + side, H)
            cells = wanted(r0, r1, c0, c1)
            if not cells[0].size:
                continue
            reg = (c0 * gsd - margin, -r1 * gsd - margin, c1 * gsd + margin, -r0 * gsd + margin)
            jobs.append(((r0, r1, c0, c1), cells, reg))
    hull = None
    while jobs:
        def run(j):
            blk, cells, reg = j
            whole = reg is None or (reg[0] <= ext[0] and reg[2] >= ext[2] and reg[1] <= ext[1] and reg[3] >= ext[3])
            if whole:
                return _tin_cells(dtm, grid, lx, ly, z, blk, cells, ext, None)
            return _tin_cells(dtm, grid, *gather(reg), blk, cells, ext, reg)
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(run, jobs))
        jobs = []
        if hull is None:
            # every vertex of the full hull is a hull vertex of some block: hull of their union
            hv = [r["hull"] for r in res if r["hull"] is not None]
            hull = _Hull(np.concatenate(hv) if hv else np.zeros((0, 2)))
        for r in res:
            if r["redo"] is None:
                continue
            rows, cols, outside, (x0, y0, x1, y1) = r["redo"]
            if outside.any():               # outside this block's TIN: fine if outside the full hull too
                inh = hull.contains((cols + 0.5) * gsd, -(rows + 0.5) * gsd)
                keep = ~outside | inh
                rows, cols = rows[keep], cols[keep]
                if not rows.size:
                    continue
            jobs.append(((0, H, 0, W), (rows, cols), (x0, y0, x1, y1)))
        # retries that need every point: one triangulation for all of them, not one per block
        covers = lambda r: r[0] <= ext[0] and r[2] >= ext[2] and r[1] <= ext[1] and r[3] >= ext[3]
        whole = [j for j in jobs if covers(j[2])]
        if len(whole) > 1:
            jobs = [j for j in jobs if not covers(j[2])]
            jobs.append(((0, H, 0, W), (np.concatenate([j[1][0] for j in whole]),
                                        np.concatenate([j[1][1] for j in whole])), None))
    return dtm, np.isfinite(dtm)


class _Bins:
    """Points sorted into square bins once, so a region's points are found
    without scanning them all (int32 index: 4 bytes per point)."""

    def __init__(self, lx, ly, size, nx, ny):
        self.size, self.nx, self.ny = size, nx, ny
        bx = np.clip(np.floor(lx / size), 0, nx - 1).astype(np.int32)
        by = np.clip(np.floor(-ly / size), 0, ny - 1).astype(np.int32)
        b = by * nx + bx
        del bx, by
        self.order = np.argsort(b, kind="stable").astype(np.int32 if b.size < 2 ** 31 else np.int64)
        self.start = np.searchsorted(b[self.order], np.arange(nx * ny + 1))
        self.lx, self.ly = lx, ly

    def query(self, reg):
        x0, y0, x1, y1 = reg
        s = self.size
        cx0, cx1 = int(np.clip(np.floor(x0 / s), 0, self.nx - 1)), int(np.clip(np.floor(x1 / s), 0, self.nx - 1))
        cy0, cy1 = int(np.clip(np.floor(-y1 / s), 0, self.ny - 1)), int(np.clip(np.floor(-y0 / s), 0, self.ny - 1))
        parts = [self.order[self.start[r * self.nx + cx0]:self.start[r * self.nx + cx1 + 1]]
                 for r in range(cy0, cy1 + 1)]
        i = np.sort(np.concatenate(parts)) if parts else np.zeros(0, np.int64)   # keep file order
        px, py = self.lx[i], self.ly[i]
        return i[(px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)]


class _Hull:
    """Point-in-convex-hull test for a (small) set of hull candidate points."""

    def __init__(self, pts):
        from scipy.spatial import Delaunay
        self.d = None
        if len(pts) >= 3:
            try:
                self.d = Delaunay(np.unique(pts, axis=0))
            except Exception:
                pass

    def contains(self, x, y):
        if self.d is None:
            return np.ones(np.size(x), bool)          # unknown: treat as inside (recomputed exactly)
        return self.d.find_simplex(np.column_stack([x, y])) >= 0


def _triangulate(pts: np.ndarray) -> np.ndarray:
    """Delaunay triangles [m, 3] (indices into pts). Shewchuk's Triangle when
    installed (~10x faster than Qhull on lidar), else scipy (Qhull)."""
    try:
        import triangle
    except ImportError:
        triangle = None
    if triangle is not None:
        try:
            return np.asarray(triangle.triangulate({"vertices": pts}, "Q")["triangles"], np.int64)
        except Exception:
            pass
    from scipy.spatial import Delaunay
    return Delaunay(pts).simplices


def _locate(pts, tris, rr, cc, gsd, max_items: int = 4_000_000):
    """For cells (rr, cc): the triangle holding each cell centre and its
    barycentric weights. Each triangle is scanned row by row (the x span of
    the triangle at each cell-centre row, like a scanline rasteriser), so
    the work is the number of cells inside the triangles plus their rows,
    also for the long thin triangles spanning a void (a bounding-box scan
    would be quadratic there). Processed in chunks of at most max_items.
    Returns (index into rr/cc, triangle index, weights [k, 3])."""
    r0, r1, c0, c1 = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
    want = np.full((r1 - r0, c1 - c0), -1, np.int64)
    want[rr - r0, cc - c0] = np.arange(rr.size)
    V = pts[tris]                                               # [m, 3, 2]
    ry0 = np.maximum(np.ceil(-V[:, :, 1].max(1) / gsd - 0.5), r0).astype(np.int64)
    ry1 = np.minimum(np.floor(-V[:, :, 1].min(1) / gsd - 0.5), r1 - 1).astype(np.int64)
    cx0 = np.maximum(np.ceil(V[:, :, 0].min(1) / gsd - 0.5), c0)
    cx1 = np.minimum(np.floor(V[:, :, 0].max(1) / gsd - 0.5), c1 - 1)
    t_all = np.flatnonzero((ry1 >= ry0) & (cx1 >= cx0))
    out_c, out_t, out_w = [], [], []
    chunk = np.cumsum(ry1[t_all] - ry0[t_all] + 1) // max_items
    edges = np.flatnonzero(np.diff(chunk)) + 1
    for t in np.split(t_all, edges):
        if not t.size:
            continue
        # one item per (triangle, row): the triangle's x span on that row's centre line
        nr = ry1[t] - ry0[t] + 1
        ti = np.repeat(np.arange(t.size), nr)
        row = ry0[t][ti] + (np.arange(nr.sum()) - np.repeat(np.cumsum(nr) - nr, nr))
        yc = -(row + 0.5) * gsd
        Vt = V[t][ti]                                           # [k, 3, 2]
        xl = np.full(yc.size, np.inf)
        xr = np.full(yc.size, -np.inf)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            xa, ya, xb, yb = Vt[:, a, 0], Vt[:, a, 1], Vt[:, b, 0], Vt[:, b, 1]
            cross = (np.minimum(ya, yb) <= yc) & (yc <= np.maximum(ya, yb)) & (ya != yb)
            with np.errstate(divide="ignore", invalid="ignore"):
                xi = xa + (yc - ya) * (xb - xa) / (yb - ya)
            xl = np.where(cross, np.minimum(xl, xi), xl)
            xr = np.where(cross, np.maximum(xr, xi), xr)
            flat = (ya == yb) & (ya == yc)                      # an edge lying on the line
            xl = np.where(flat, np.minimum(xl, np.minimum(xa, xb)), xl)
            xr = np.where(flat, np.maximum(xr, np.maximum(xa, xb)), xr)
        ca = np.maximum(np.ceil(xl / gsd - 0.5 - 1e-9), c0)
        cb = np.minimum(np.floor(xr / gsd - 0.5 + 1e-9), c1 - 1)
        ok = np.isfinite(ca) & np.isfinite(cb) & (cb >= ca)
        ti, row, ca, cb = ti[ok], row[ok], ca[ok].astype(np.int64), cb[ok].astype(np.int64)
        ncol = cb - ca + 1
        k = np.repeat(np.arange(ti.size), ncol)
        col = ca[k] + (np.arange(ncol.sum()) - np.repeat(np.cumsum(ncol) - ncol, ncol))
        row, ti = row[k], ti[k]
        cell = want[row - r0, col - c0]
        m = cell >= 0
        cell, row, col, tri = cell[m], row[m], col[m], t[ti[m]]
        a, b, c = V[tri, 0], V[tri, 1], V[tri, 2]
        px, py = (col + 0.5) * gsd, -(row + 0.5) * gsd
        den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            l1 = ((b[:, 1] - c[:, 1]) * (px - c[:, 0]) + (c[:, 0] - b[:, 0]) * (py - c[:, 1])) / den
            l2 = ((c[:, 1] - a[:, 1]) * (px - c[:, 0]) + (a[:, 0] - c[:, 0]) * (py - c[:, 1])) / den
        l3 = 1.0 - l1 - l2
        eps = -1e-9
        inside = np.isfinite(l1) & np.isfinite(l2) & (l1 >= eps) & (l2 >= eps) & (l3 >= eps)
        out_c.append(cell[inside])
        out_t.append(tri[inside])
        out_w.append(np.column_stack([l1[inside], l2[inside], l3[inside]]))
    if not out_c:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3))
    cell, tri, w = np.concatenate(out_c), np.concatenate(out_t), np.concatenate(out_w)
    # a centre on a shared edge: either triangle (same value); prefer the one it is most inside
    order = np.lexsort((-w.min(1), cell))
    cell, tri, w = cell[order], tri[order], w[order]
    first = np.ones(cell.size, bool)
    first[1:] = cell[1:] != cell[:-1]
    return cell[first], tri[first], w[first]


def _tin_cells(dtm, grid, px, py, pz, blk, cells, ext, region=None):
    """Fill dtm cells of block blk = (r0, r1, c0, c1) (only `cells` = (rows,
    cols) if given) from a TIN of the points px/py/pz. region: the extent
    the points were taken from (None = all points: every value is final).
    Returns {"hull": this TIN's hull vertices, "redo": None or (rows, cols,
    outside_tin, region to redo them with)}: cells whose triangle may depend
    on points outside the region, or that fall outside this TIN."""
    r0, r1, c0, c1 = blk
    if cells is None:
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
        rr, cc = rr.ravel(), cc.ravel()
    else:
        rr, cc = cells
    tris = None
    if px.size >= 3:
        # points sharing x, y (different z): keep the lowest, so every block makes the same choice
        o = np.lexsort((pz, py, px))
        px, py, pz = px[o], py[o], pz[o]
        first = np.ones(px.size, bool)
        first[1:] = (px[1:] != px[:-1]) | (py[1:] != py[:-1])
        if not first.all():
            px, py, pz = px[first], py[first], pz[first]
        pts = np.column_stack([px, py]).astype(np.float64)
        try:
            tris = _triangulate(pts)
        except Exception:                               # e.g. all points on a line
            tris = None
        if tris is not None and not len(tris):
            tris = None
    if tris is None:
        if region is None:
            return {"hull": None, "redo": None}
        # no TIN here (no points, e.g. the buffer beyond the last tile): like cells outside a
        # TIN, fine if outside the full hull, else redone with a wider region
        x0, y0, x1, y1 = region
        g = max(30.0, 0.3 * max(x1 - x0, y1 - y0))
        return {"hull": None, "redo": (rr, cc, np.ones(rr.size, bool), (x0 - g, y0 - g, x1 + g, y1 + g))}
    ci, ti, w = _locate(pts, tris, rr, cc, grid.gsd)
    val = (np.asarray(pz, np.float64)[tris[ti]] * w).sum(1)
    try:
        from scipy.spatial import ConvexHull
        hull = pts[ConvexHull(pts).vertices]
    except Exception:
        hull = pts
    inside = np.zeros(rr.size, bool)
    inside[ci] = True
    if region is None:
        dtm[rr[ci], cc[ci]] = val
        return {"hull": hull, "redo": None}
    # accept a cell if its triangle's circumcircle holds no point the block did not see
    V = pts[tris[ti]]                                   # [k, 3, 2]
    ax, ay = V[:, 0, 0], V[:, 0, 1]
    bx, by = V[:, 1, 0] - ax, V[:, 1, 1] - ay
    cx, cy = V[:, 2, 0] - ax, V[:, 2, 1] - ay
    dd = 2.0 * (bx * cy - by * cx)
    with np.errstate(divide="ignore", invalid="ignore"):
        ux = (cy * (bx * bx + by * by) - by * (cx * cx + cy * cy)) / dd
        uy = (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by)) / dd
        rad = np.hypot(ux, uy) * (1 + 1e-9) + 1e-9
        ux, uy = ux + ax, uy + ay
        # the circle's bounding box, cut to where points exist, must lie in the region
        bx0, bx1 = np.maximum(ux - rad, ext[0]), np.minimum(ux + rad, ext[2])
        by0, by1 = np.maximum(uy - rad, ext[1]), np.minimum(uy + rad, ext[3])
    x0, y0, x1, y1 = region
    ok = np.isfinite(rad) & (bx0 >= x0) & (bx1 <= x1) & (by0 >= y0) & (by1 <= y1)
    ir, ic = rr[ci], cc[ci]
    dtm[ir[ok], ic[ok]] = val[ok]
    bad = ~ok
    if not bad.any() and inside.all():
        return {"hull": hull, "redo": None}
    rows = np.concatenate([ir[bad], rr[~inside]])
    cols = np.concatenate([ic[bad], cc[~inside]])
    outside = np.concatenate([np.zeros(int(bad.sum()), bool), np.ones(int((~inside).sum()), bool)])
    # next region: grown step by step (thin triangles at a void have huge circles), but never
    # beyond the circles that failed; degenerate triangles: the step alone
    fin = bad & np.isfinite(rad)
    g = max(30.0, 0.3 * max(x1 - x0, y1 - y0))
    if fin.any() and not (bad & ~np.isfinite(rad)).any() and inside.all():
        nreg = (max(x0 - g, float(bx0[fin].min()) - 1.0), max(y0 - g, float(by0[fin].min()) - 1.0),
                min(x1 + g, float(bx1[fin].max()) + 1.0), min(y1 + g, float(by1[fin].max()) + 1.0))
    else:
        nreg = (x0 - g, y0 - g, x1 + g, y1 + g)
    return {"hull": hull, "redo": (rows, cols, outside, nreg)}


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
        dtm_b, b_valid = tin_dtm(grid, pts.x[bg], pts.y[bg], pts.z[bg], need=survey)
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
    gt, gt_valid = tin_dtm(grid, after.x[g], after.y[g], after.z[g], need=np.asarray(survey, bool))
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


def target_from_rasters(grid: Grid, paths: list, survey: np.ndarray, flat_min_cells: int = 50,
                        keep: np.ndarray | None = None) -> dict:
    """Target DTM from published DTM rasters (the EA product), sampled at
    our cell centres (exact copy when the grids coincide, e.g. gsd 1 m on
    the whole-metre OS grid). Hydro-flattened water (flat_areas) is excluded
    from the target: its level is a production choice the points cannot show.
    keep: cells kept even when flat (under and next to bridges, where the
    model must learn to take the deck out down to the water)."""
    from ..io_raster import sample_rasters

    xs, ys = grid.cell_centres()
    gt = sample_rasters(paths, xs, ys)
    ok = np.isfinite(gt) & survey
    flat = flat_areas(gt, flat_min_cells) if flat_min_cells else np.zeros_like(ok)
    ok &= ~(flat & ~keep) if keep is not None else ~flat
    return {"gt_dtm": np.where(ok, gt, np.nan).astype(np.float32), "gt_valid": ok.astype(np.float32),
            "flat_water": flat.astype(np.float32)}


def line_mask(grid: Grid, lines: list, half_width: float) -> np.ndarray:
    """Cells within half_width metres of any polyline ([[x, y], ...], grid CRS)."""
    from scipy.ndimage import distance_transform_edt

    mark = np.zeros((grid.height, grid.width), bool)
    pad = half_width + grid.gsd
    x0, x1 = grid.xmin - pad, grid.xmin + grid.width * grid.gsd + pad
    y1, y0 = grid.ymax + pad, grid.ymax - grid.height * grid.gsd - pad
    for line in lines:
        a = np.asarray(line, np.float64).reshape(-1, 2)
        if len(a) == 1:
            a = np.vstack([a, a])
        if a[:, 0].max() < x0 or a[:, 0].min() > x1 or a[:, 1].max() < y0 or a[:, 1].min() > y1:
            continue
        for (xa, ya), (xb, yb) in zip(a[:-1], a[1:]):
            n = max(2, int(np.hypot(xb - xa, yb - ya) / (grid.gsd / 2)) + 2)
            t = np.linspace(0, 1, n)
            c = np.floor((xa + t * (xb - xa) - grid.xmin) / grid.gsd).astype(np.int64)
            r = np.floor((grid.ymax - (ya + t * (yb - ya))) / grid.gsd).astype(np.int64)
            ok = (r >= 0) & (r < grid.height) & (c >= 0) & (c < grid.width)
            mark[r[ok], c[ok]] = True
    if not mark.any():
        return mark
    return distance_transform_edt(~mark) * grid.gsd <= half_width


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

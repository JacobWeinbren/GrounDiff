"""Rasterise LAS/LAZ/COPC files in one pass with bounded memory.

The same rasters as rasterise.input_rasters, but the points are never all
held in memory: each chunk of a file is folded into per-cell accumulators
(count, min / max / lowest last return, running mean and M2 for z_std,
echo sum, ground / non-ground counts) and dropped. Only the lasground_new
ground points are kept, as float32 coordinates relative to the grid
(12 bytes per point), for the blocked TIN (rasterise.tin_dtm_local).

Memory: about 50 bytes per cell (a 5 km tile with its buffer at 1 m:
~1.5 GB) + 12 bytes per ground point + the TIN blocks, instead of
~550 bytes per point for the whole tile.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .laz import _crs_wkt, _from_record, _is_copc, header_info
from .rasterise import Grid, _flat_index, tin_dtm_local


class Accumulator:
    def __init__(self, grid: Grid, keep_ground: bool = False, ground_classes=(2,)):
        n = grid.width * grid.height
        self.grid, self.n = grid, n
        self.count = np.zeros(n, np.int32)
        self.zmax = np.full(n, -np.inf, np.float32)
        self.zmin = np.full(n, np.inf, np.float32)
        self.zlast = np.full(n, np.inf, np.float32)
        self.mean = np.zeros(n, np.float64)
        self.m2 = np.zeros(n, np.float64)
        self.echo = np.zeros(n, np.float64)
        self.hist = np.zeros(256, np.int64)
        self.keep_ground = keep_ground
        self.ground_classes = np.asarray(ground_classes, np.uint8)
        if keep_ground:
            self.ng = np.zeros(n, np.int32)
            self.gx, self.gy, self.gz = [], [], []
        self.n_points = 0
        self.crs_wkt = None

    def add(self, pts) -> None:
        if pts.crs_wkt and not self.crs_wkt:
            self.crs_wkt = pts.crs_wkt
        if not len(pts):
            return
        self.hist += np.bincount(pts.cls, minlength=256)[:256]
        idx, ok = _flat_index(self.grid, pts.x, pts.y)
        if not ok.all():
            pts = pts.subset(ok)
            idx = idx[ok]
        if not idx.size:
            return
        self.n_points += idx.size
        z = pts.z
        u, inv = np.unique(idx, return_inverse=True)
        inv = inv.ravel()
        cb = np.bincount(inv).astype(np.float64)
        mb = np.bincount(inv, weights=z) / cb
        dev = z - mb[inv]
        m2b = np.bincount(inv, weights=dev * dev)
        na = self.count[u].astype(np.float64)
        tot = na + cb
        delta = mb - self.mean[u]
        self.mean[u] += delta * cb / tot              # Chan et al. merge: exact for any chunking
        self.m2[u] += m2b + delta * delta * na * cb / tot
        self.count[u] += cb.astype(np.int32)
        k = u.size
        t = np.full(k, -np.inf)
        np.maximum.at(t, inv, z)
        self.zmax[u] = np.maximum(self.zmax[u], t.astype(np.float32))
        t = np.full(k, np.inf)
        np.minimum.at(t, inv, z)
        self.zmin[u] = np.minimum(self.zmin[u], t.astype(np.float32))
        last = pts.return_number >= pts.number_of_returns       # last or only return
        if last.any():
            t = np.full(k, np.inf)
            np.minimum.at(t, inv[last], z[last])
            self.zlast[u] = np.minimum(self.zlast[u], t.astype(np.float32))
        self.echo[u] += np.bincount(inv, weights=pts.number_of_returns.astype(np.float64), minlength=k)
        if self.keep_ground:
            g = np.isin(pts.cls, self.ground_classes)
            self.ng[u] += np.bincount(inv, weights=g.astype(np.float64), minlength=k).astype(np.int32)
            if g.any():
                self.gx.append((pts.x[g] - self.grid.xmin).astype(np.float32))
                self.gy.append((pts.y[g] - self.grid.ymax).astype(np.float32))
                self.gz.append(pts.z[g].astype(np.float32))

    def class_histogram(self) -> dict[int, int]:
        return {int(c): int(v) for c, v in enumerate(self.hist) if v}

    def rasters(self, lasground: bool, coverage_close_m: float = 30.0) -> dict:
        """input_rasters' output for everything added."""
        from ..normalise import coverage_mask

        g = self.grid
        shp = (g.height, g.width)
        cnt = self.count
        has = cnt > 0
        c = np.maximum(cnt, 1)
        out = {
            "dsm_max": np.where(has, self.zmax, np.nan).astype(np.float32).reshape(shp),
            "dsm_min": np.where(has, self.zmin, np.nan).astype(np.float32).reshape(shp),
            "density": (cnt / (g.gsd * g.gsd)).astype(np.float32).reshape(shp),
            "z_std": np.where(has, np.sqrt(self.m2 / c), 0.0).astype(np.float32).reshape(shp),
            "has_return": has.astype(np.float32).reshape(shp),
            "dsm_last": np.where(np.isfinite(self.zlast), self.zlast, np.nan).astype(np.float32).reshape(shp),
            "echoes": np.where(has, self.echo / c, 0.0).astype(np.float32).reshape(shp),
        }
        del self.zmax, self.zmin, self.zlast, self.mean, self.m2, self.echo
        survey = coverage_mask(has.reshape(shp), g.gsd, coverage_close_m)
        out["in_survey"] = survey.astype(np.float32)
        if lasground and self.keep_ground:
            gx = np.concatenate(self.gx) if self.gx else np.zeros(0, np.float32)
            self.gx = []
            gy = np.concatenate(self.gy) if self.gy else np.zeros(0, np.float32)
            self.gy = []
            gz = np.concatenate(self.gz) if self.gz else np.zeros(0, np.float32)
            self.gz = []
            dtm_b, b_valid = tin_dtm_local(g, gx, gy, gz, need=survey)
            del gx, gy, gz
            b_valid &= survey
            out["dtm_before"] = np.where(b_valid, dtm_b, np.nan).astype(np.float32)
            out["before_valid"] = b_valid.astype(np.float32)
            ground = has & (self.ng >= cnt - self.ng)              # mode label, ties to ground
            out["sem_ground"] = ground.astype(np.float32).reshape(shp)
            out["sem_nonground"] = (has & ~ground).astype(np.float32).reshape(shp)
        return out


def copc_twin(path: Path) -> Path | None:
    """The .copc.laz QGIS writes next to a LAS/LAZ it has loaded, if it is
    complete (same point count): lets a buffer strip be read without
    decompressing the whole neighbouring file."""
    path = Path(path)
    if path.name.lower().endswith(".copc.laz"):
        return None
    twin = path.with_name(path.stem + ".copc.laz")
    if not twin.exists():
        return None
    try:
        a, b = header_info(path), header_info(twin)
        return twin if b["is_copc"] and a["point_count"] == b["point_count"] else None
    except Exception:
        return None


def stream_file(path, bbox, acc: Accumulator, read_opts: dict | None = None, chunk_size: int = 1_000_000,
                progress: Callable | None = None, cancelled: Callable = lambda: False) -> None:
    """Fold the points of path inside bbox (xmin, ymin, xmax, ymax; half open)
    into acc. progress(fraction of this file) is called per chunk;
    cancelled() is checked per chunk (raises RuntimeError('cancelled'))."""
    import laspy

    ro = read_opts or {}
    opts = (tuple(ro.get("drop_classes", ())), ro.get("drop_withheld", True), ro.get("drop_overlap", False),
            ro.get("drop_synthetic", False))
    path = Path(path)
    src = path
    with laspy.open(str(path)) as f:
        is_copc = _is_copc(f.header)
        total = max(int(f.header.point_count), 1)
        hb = (float(f.header.mins[0]), float(f.header.mins[1]), float(f.header.maxs[0]), float(f.header.maxs[1]))
    if not is_copc:
        twin = copc_twin(path)
        if twin is not None:
            src, is_copc = twin, True
    inter = (max(bbox[0], hb[0]), max(bbox[1], hb[1]), min(bbox[2], hb[2] + 1e-6), min(bbox[3], hb[3] + 1e-6))
    if inter[0] >= inter[2] or inter[1] >= inter[3]:
        return
    frac_needed = (inter[2] - inter[0]) * (inter[3] - inter[1]) / max((hb[2] - hb[0]) * (hb[3] - hb[1]), 1e-9)
    if is_copc and frac_needed < 0.999:
        before = acc.n_points
        try:
            _stream_copc(src, inter, acc, opts, progress, cancelled)
            return
        except RuntimeError as e:
            if str(e) == "cancelled":
                raise
            if acc.n_points != before:
                raise
        except Exception:
            if acc.n_points != before:            # partly added: cannot fall back without double counting
                raise
        # no COPC support (e.g. lazrs missing) or an unfinished QGIS index: read the file itself
    done = 0
    with laspy.open(str(path)) as f:
        crs = _crs_wkt(f.header)
        for rec in f.chunk_iterator(chunk_size):
            if cancelled():
                raise RuntimeError("cancelled")
            done += len(rec)
            acc.add(_from_record(rec, crs, *opts, bbox))
            if progress:
                progress(min(done / total, 1.0))


def _stream_copc(src, inter, acc, opts, progress, cancelled):
    """Octree queries over ~500 m windows: only the needed nodes are decompressed, and each
    window's points are folded in and dropped before the next."""
    import laspy
    from laspy.copc import Bounds
    step = 500.0
    xs = np.arange(inter[0], inter[2], step).tolist() + [inter[2]]
    ys = np.arange(inter[1], inter[3], step).tolist() + [inter[3]]
    wins = [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for i in range(len(xs) - 1) for j in range(len(ys) - 1)]
    with laspy.CopcReader.open(str(src)) as r:
        crs = _crs_wkt(r.header)
        for k, w in enumerate(wins):
            if cancelled():
                raise RuntimeError("cancelled")
            rec = r.query(bounds=Bounds(mins=np.array(w[:2], float), maxs=np.array(w[2:], float)))
            acc.add(_from_record(rec, crs, *opts, w))   # half open: shared edges counted once
            if progress:
                progress((k + 1) / len(wins))


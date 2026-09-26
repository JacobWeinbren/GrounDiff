"""Read LAS/LAZ/COPC point clouds with laspy.

Tolerant of producer differences (EA deliveries vs files saved by editing
software such as LP360): LAS 1.2-1.4, point formats 0-10, missing WKT bit,
extra bytes, GeoTIFF-key or WKT CRS, inconsistent return numbers. Run
`python -m groundiff.data.lasinspect` to see what a file contains.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ASPRS: 7 = low point (noise), 18 = high noise. Excluded from every raster.
NOISE_CLASSES = (7, 18)
LEGACY_OVERLAP_CLASS = 12


@dataclass
class Points:
    x: np.ndarray            # float64
    y: np.ndarray
    z: np.ndarray
    cls: np.ndarray          # uint8
    return_number: np.ndarray
    number_of_returns: np.ndarray
    crs_wkt: str | None

    def __len__(self):
        return self.x.size

    def subset(self, mask: np.ndarray) -> "Points":
        return Points(self.x[mask], self.y[mask], self.z[mask], self.cls[mask],
                      self.return_number[mask], self.number_of_returns[mask], self.crs_wkt)


def _crs_wkt(header) -> str | None:
    try:
        crs = header.parse_crs()
        return crs.to_wkt() if crs is not None else None
    except Exception:
        return None


def _read(path: Path):
    """laspy.read, retrying other LAZ backends if the default one fails."""
    import laspy
    try:
        return laspy.read(str(path))
    except Exception as first:
        if path.suffix.lower() != ".laz":
            raise
        for backend in getattr(laspy, "LazBackend", []):
            try:
                if backend.is_available():
                    return laspy.read(str(path), laz_backend=backend)
            except Exception:
                continue
        raise RuntimeError(f"could not read {path}: {first!r} (is lazrs or laszip installed?)") from first


def _flag(las, name):
    try:
        return np.asarray(getattr(las, name), dtype=bool)
    except Exception:
        return None


def _from_record(rec, crs_wkt, drop_classes, drop_withheld, drop_overlap, drop_synthetic, bbox=None) -> Points:
    x = np.asarray(rec.x, dtype=np.float64)
    y = np.asarray(rec.y, dtype=np.float64)
    z = np.asarray(rec.z, dtype=np.float64)
    cls = np.asarray(rec.classification, dtype=np.uint8)
    rn = np.asarray(rec.return_number, dtype=np.uint8)
    nr = np.asarray(rec.number_of_returns, dtype=np.uint8)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if bbox is not None:
        keep &= (x >= bbox[0]) & (x < bbox[2]) & (y >= bbox[1]) & (y < bbox[3])
    if drop_classes:
        keep &= ~np.isin(cls, np.asarray(drop_classes, dtype=np.uint8))
    for name, on in (("withheld", drop_withheld), ("synthetic", drop_synthetic), ("overlap", drop_overlap)):
        fl = _flag(rec, name) if on else None
        if fl is not None:
            keep &= ~fl
    if drop_overlap:
        keep &= cls != LEGACY_OVERLAP_CLASS
    # number_of_returns = 0 or return_number = 0 appear in some files: treat as single returns;
    # return_number > number_of_returns is treated as a last return by the rasteriser (rn >= nr)
    pts = Points(x, y, z, cls, np.maximum(rn, 1), np.maximum(nr, 1), crs_wkt)
    return pts if keep.all() else pts.subset(keep)


def read_points(path: str | Path, drop_classes=NOISE_CLASSES, drop_withheld: bool = True,
                drop_overlap: bool = False, drop_synthetic: bool = False) -> Points:
    path = Path(path)
    las = _read(path)
    pts = _from_record(las, _crs_wkt(las.header), drop_classes, drop_withheld, drop_overlap, drop_synthetic)
    if len(pts) == 0:
        raise ValueError(f"no usable points in {path}")
    return pts


def header_bounds(path: str | Path) -> tuple[float, float, float, float]:
    import laspy
    with laspy.open(str(path)) as f:
        mn, mx = f.header.mins, f.header.maxs
    return float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])


def read_points_bbox(path: str | Path, bbox, drop_classes=NOISE_CLASSES, drop_withheld: bool = True,
                     drop_overlap: bool = False, drop_synthetic: bool = False,
                     chunk_size: int = 2_000_000) -> Points:
    """Points with xmin <= x < xmax, ymin <= y < ymax, read in chunks so that
    only the kept points are held in memory. May return 0 points."""
    import laspy
    parts = []
    with laspy.open(str(path)) as f:
        crs = _crs_wkt(f.header)
        for rec in f.chunk_iterator(chunk_size):
            pts = _from_record(rec, crs, drop_classes, drop_withheld, drop_overlap, drop_synthetic, bbox)
            if len(pts):
                parts.append(pts)
    return concat(parts, crs)


def concat(parts: list, crs_wkt: str | None = None) -> Points:
    if not parts:
        e = np.empty(0)
        u = np.empty(0, np.uint8)
        return Points(e, e.copy(), e.copy(), u, u.copy(), u.copy(), crs_wkt)
    return Points(*(np.concatenate([getattr(p, k) for p in parts]) for k in
                    ("x", "y", "z", "cls", "return_number", "number_of_returns")),
                  crs_wkt or next((p.crs_wkt for p in parts if p.crs_wkt), None))


def class_histogram(cls: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(cls, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}

"""Read LAS/LAZ/COPC point clouds with laspy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ASPRS: 7 = low point (noise), 18 = high noise. Excluded from every raster.
NOISE_CLASSES = (7, 18)


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


def read_points(path: str | Path, drop_classes=NOISE_CLASSES, drop_withheld: bool = True) -> Points:
    import laspy

    las = laspy.read(str(path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    cls = np.asarray(las.classification, dtype=np.uint8)
    rn = np.asarray(las.return_number, dtype=np.uint8)
    nr = np.asarray(las.number_of_returns, dtype=np.uint8)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if drop_classes:
        keep &= ~np.isin(cls, np.asarray(drop_classes, dtype=np.uint8))
    if drop_withheld:
        try:
            keep &= ~np.asarray(las.withheld, dtype=bool)
        except Exception:
            pass
    # some files carry number_of_returns = 0; treat as single returns
    nr = np.maximum(nr, 1)
    rn = np.clip(rn, 1, None)
    pts = Points(x, y, z, cls, rn, nr, _crs_wkt(las.header))
    if not keep.all():
        pts = pts.subset(keep)
    if len(pts) == 0:
        raise ValueError(f"no usable points in {path}")
    return pts


def class_histogram(cls: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(cls, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}

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

# Class codes. EA production editing (lasground_new, then LP360) uses only
# 1 unclassified, 2 ground, bridge and 7 low noise, and builds the DTM from
# ground. The classes in the published EA LAZ/COPC files come from a different
# process (they include 3-6 vegetation/buildings) and do not match the
# published DTM rasters, so they are never used: input rasters take every
# point, and only lasground_new's own 1/2 split feeds dtm_before / sem_*.
# NOISE_CLASSES is only for reading hand-edited tiles (drop_classes=...).
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


PROJECTED_CS_KEY, GEOGRAPHIC_CS_KEY = 3072, 2048
WKT_RECORD, GEOKEY_RECORD = 2112, 34735


def _crs_vlrs(header) -> list:
    vlrs = list(getattr(header, "vlrs", []) or []) + list(getattr(header, "evlrs", None) or [])
    return [v for v in vlrs if getattr(v, "user_id", "").rstrip("\0") == "LASF_Projection"
            and getattr(v, "record_id", None) in (WKT_RECORD, GEOKEY_RECORD)]


def crs_from_header(header) -> tuple[str | None, str | None]:
    """(WKT or None, note or None). Uses pyproj through laspy when available,
    otherwise reads the WKT VLR text or the GeoTIFF EPSG key directly (QGIS's
    Python often lacks pyproj). VLRs and EVLRs are both searched."""
    vlrs = _crs_vlrs(header)
    if not vlrs:
        return None, None
    try:
        crs = header.parse_crs()
        if crs is not None:
            return crs.to_wkt(), None
    except Exception:
        pass
    for v in vlrs:
        if v.record_id == WKT_RECORD:
            text = getattr(v, "string", None)
            if isinstance(text, bytes):
                text = text.decode("utf-8", "replace")
            text = (text or "").strip().rstrip("\0").strip()
            if text:
                return text, None
    for v in vlrs:
        if v.record_id == GEOKEY_RECORD:
            for key in getattr(v, "geo_keys", []):
                if key.id in (PROJECTED_CS_KEY, GEOGRAPHIC_CS_KEY) and key.tiff_tag_location == 0 \
                        and 1024 <= key.value_offset < 32767:
                    from ..io_raster import crs_wkt_from_epsg
                    wkt = crs_wkt_from_epsg(int(key.value_offset))
                    if wkt:
                        return wkt, None
                    return None, f"CRS VLR names EPSG:{key.value_offset} but it could not be converted"
    return None, "file has a CRS VLR that could not be parsed"


def _crs_wkt(header) -> str | None:
    return crs_from_header(header)[0]


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


def read_points(path: str | Path, drop_classes=(), drop_withheld: bool = True,
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


def header_info(path: str | Path) -> dict:
    import laspy
    with laspy.open(str(path)) as f:
        h = f.header
        mn, mx = h.mins, h.maxs
        return {"bounds": (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])),
                "point_count": int(h.point_count), "is_copc": _is_copc(h)}


def data_bounds(path: str | Path, chunk_size: int = 2_000_000) -> tuple[float, float, float, float] | None:
    """Bounds of the points themselves (one pass over x/y), for files whose
    header bounds cannot be trusted. None if the file has no points."""
    import laspy
    b = [np.inf, np.inf, -np.inf, -np.inf]
    with laspy.open(str(path)) as f:
        for rec in f.chunk_iterator(chunk_size):
            x, y = np.asarray(rec.x), np.asarray(rec.y)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.any():
                b = [min(b[0], x[ok].min()), min(b[1], y[ok].min()), max(b[2], x[ok].max()), max(b[3], y[ok].max())]
    return None if not np.isfinite(b[0]) else tuple(float(v) for v in b)


def _is_copc(header) -> bool:
    return any(getattr(v, "user_id", "").rstrip("\0") == "copc" for v in getattr(header, "vlrs", []) or [])


def _bbox_copc(path: Path, bbox, opts) -> Points:
    import laspy
    from laspy.copc import Bounds
    with laspy.CopcReader.open(str(path)) as r:
        crs = _crs_wkt(r.header)
        rec = r.query(bounds=Bounds(mins=np.array(bbox[:2], float), maxs=np.array(bbox[2:], float)))
        return _from_record(rec, crs, *opts, bbox)          # exact half-open filter on top of the octree


def _bbox_chunks(path: Path, bbox, opts, chunk_size: int, laz_backend=None) -> Points:
    import laspy
    parts = []
    kw = {"laz_backend": laz_backend} if laz_backend is not None else {}
    with laspy.open(str(path), **kw) as f:
        crs = _crs_wkt(f.header)
        for rec in f.chunk_iterator(chunk_size):
            pts = _from_record(rec, crs, *opts, bbox)
            if len(pts):
                parts.append(pts)
    return concat(parts, crs)


def read_points_bbox(path: str | Path, bbox, drop_classes=(), drop_withheld: bool = True,
                     drop_overlap: bool = False, drop_synthetic: bool = False,
                     chunk_size: int = 2_000_000) -> Points:
    """Points with xmin <= x < xmax, ymin <= y < ymax. COPC files are queried
    through their octree (only the needed nodes are decompressed); others are
    read in chunks so only the kept points are held in memory. Falls back to
    other LAZ backends like read_points. May return 0 points. Errors name the
    file."""
    import laspy
    path = Path(path)
    opts = (drop_classes, drop_withheld, drop_overlap, drop_synthetic)
    try:
        with laspy.open(str(path)) as f:
            copc = _is_copc(f.header)
    except Exception as e:
        raise RuntimeError(f"cannot open {path.name}: {e!r}") from e
    if copc:
        try:
            return _bbox_copc(path, bbox, opts)
        except Exception:
            pass                                            # e.g. no lazrs: read it the ordinary way
    try:
        return _bbox_chunks(path, bbox, opts, chunk_size)
    except Exception as first:
        if path.suffix.lower() == ".laz":
            for backend in getattr(laspy, "LazBackend", []):
                try:
                    if backend.is_available():
                        return _bbox_chunks(path, bbox, opts, chunk_size, backend)
                except Exception:
                    continue
        raise RuntimeError(f"cannot read {path.name}: {first!r}") from first


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

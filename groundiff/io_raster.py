"""GeoTIFF read/write via rasterio, or GDAL's Python bindings (as in QGIS)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def crs_wkt_from_epsg(epsg: int = 27700) -> str | None:
    """WKT for an EPSG code via rasterio or GDAL (None if neither is available)."""
    try:
        from rasterio.crs import CRS
        return CRS.from_epsg(epsg).to_wkt()
    except ImportError:
        pass
    try:
        from osgeo import osr
        srs = osr.SpatialReference()
        if srs.ImportFromEPSG(int(epsg)) != 0:
            return None
        return srs.ExportToWkt() or None
    except (ImportError, RuntimeError):
        return None


def write_geotiff(path: str | Path, arr: np.ndarray, xmin: float, ymax: float, gsd: float,
                  crs_wkt: str | None = None, nodata: float = -9999.0):
    a = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
    try:
        import rasterio
        from rasterio.transform import from_origin
        with rasterio.open(str(path), "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1,
                           dtype="float32", crs=crs_wkt, transform=from_origin(xmin, ymax, gsd, gsd),
                           nodata=nodata, compress="deflate", tiled=True) as dst:
            dst.write(a, 1)
        return
    except ImportError:
        pass
    from osgeo import gdal, osr
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), a.shape[1], a.shape[0], 1, gdal.GDT_Float32, ["COMPRESS=DEFLATE", "TILED=YES"])
    ds.SetGeoTransform((xmin, gsd, 0.0, ymax, 0.0, -gsd))
    if crs_wkt:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(crs_wkt)
        ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(a)
    ds.FlushCache()


def _north_up(a: np.ndarray, gt: tuple, path) -> tuple[np.ndarray, tuple]:
    """gt = GDAL geotransform. Flips south-up rasters; rejects rotation and
    non-square pixels (relative tolerance 1e-6)."""
    x0, px, rx, y0, ry, py = gt
    if rx or ry or px <= 0:
        raise ValueError(f"{path}: rotated or mirrored rasters are not supported")
    if abs(abs(px) - abs(py)) > 1e-6 * abs(px):
        raise ValueError(f"{path}: only square pixels are supported ({px} x {abs(py)})")
    if py > 0:                                   # south-up: row 0 is the southern edge
        a = a[..., ::-1, :]
        y0 = y0 + a.shape[-2] * py
    return a, (x0, px, 0.0, y0, 0.0, -abs(px))


def read_geotiff(path: str | Path) -> tuple[np.ndarray, dict]:
    """Returns (array with NaN for no-data, info{xmin, ymax, gsd, crs_wkt}).
    Reads anything GDAL reads (GeoTIFF, ASCII grid, VRT...)."""
    try:
        import rasterio
        with rasterio.open(str(path)) as src:
            a = src.read(1).astype(np.float64)
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            gt = src.transform.to_gdal()
            crs = src.crs.to_wkt() if src.crs else None
    except ImportError:
        from osgeo import gdal
        ds = gdal.Open(str(path))
        if ds is None:
            raise ValueError(f"cannot open {path}")
        band = ds.GetRasterBand(1)
        a = band.ReadAsArray().astype(np.float64)
        nd = band.GetNoDataValue()
        if nd is not None:
            a[a == nd] = np.nan
        gt = ds.GetGeoTransform()
        crs = ds.GetProjection() or None
    a[~(np.abs(a) < 1e30)] = np.nan          # EA rasters use -3.4e38 (float32 min) as nodata
    a, gt = _north_up(a, gt, path)
    return a, {"xmin": gt[0], "ymax": gt[3], "gsd": gt[1], "crs_wkt": crs}


def raster_info(path: str | Path) -> dict:
    """Extent and pixel size without reading the data."""
    try:
        import rasterio
        with rasterio.open(str(path)) as src:
            gt, w, h = src.transform.to_gdal(), src.width, src.height
    except ImportError:
        from osgeo import gdal
        ds = gdal.Open(str(path))
        if ds is None:
            raise ValueError(f"cannot open {path}")
        gt, w, h = ds.GetGeoTransform(), ds.RasterXSize, ds.RasterYSize
    x0, px, _, y0, _, py = gt
    ys = sorted((y0, y0 + h * py))
    return {"xmin": x0, "xmax": x0 + w * px, "ymin": ys[0], "ymax": ys[1], "res": abs(px),
            "width": w, "height": h}


def _read_window(path, xmin, ymin, xmax, ymax):
    """Pixels of `path` covering the bbox (plus one pixel), as (array NaN=no-data,
    x0 of the window's left edge, y0 of its top edge, res)."""
    info = raster_info(path)
    res = info["res"]
    c0 = max(0, int(np.floor((xmin - info["xmin"]) / res)) - 1)
    c1 = min(info["width"], int(np.ceil((xmax - info["xmin"]) / res)) + 1)
    r0 = max(0, int(np.floor((info["ymax"] - ymax) / res)) - 1)
    r1 = min(info["height"], int(np.ceil((info["ymax"] - ymin) / res)) + 1)
    if c1 <= c0 or r1 <= r0:
        return None
    try:
        import rasterio
        from rasterio.windows import Window
        with rasterio.open(str(path)) as src:
            flip = src.transform.e > 0
            rr0 = src.height - r1 if flip else r0
            a = src.read(1, window=Window(c0, rr0, c1 - c0, r1 - r0)).astype(np.float64)
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
    except ImportError:
        from osgeo import gdal
        ds = gdal.Open(str(path))
        flip = ds.GetGeoTransform()[5] > 0
        rr0 = ds.RasterYSize - r1 if flip else r0
        band = ds.GetRasterBand(1)
        a = band.ReadAsArray(c0, rr0, c1 - c0, r1 - r0).astype(np.float64)
        nd = band.GetNoDataValue()
        if nd is not None:
            a[a == nd] = np.nan
    if flip:
        a = a[::-1]
    a[~(np.abs(a) < 1e30)] = np.nan          # EA rasters use -3.4e38 (float32 min) as nodata
    return a, info["xmin"] + c0 * res, info["ymax"] - r0 * res, res


def _mosaic_windows(paths: list, xmin, ymin, xmax, ymax):
    """Windows of the rasters over the bbox, with rasters that share a pixel
    grid (same size and alignment, e.g. abutting 5 km EA tiles) merged into
    one array so interpolation does not stop at tile edges. Earlier paths win."""
    groups: dict = {}
    order = []
    for p in paths:
        got = _read_window(p, xmin, ymin, xmax, ymax)
        if got is None:
            continue
        a, x0, y0, res = got
        key = (round(res, 9), round((x0 / res) % 1, 6), round((y0 / res) % 1, 6))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(got)
    for key in order:
        parts = groups[key]
        res = parts[0][3]
        gx0 = min(p[1] for p in parts)
        gy0 = max(p[2] for p in parts)
        gx1 = max(p[1] + p[0].shape[1] * res for p in parts)
        gy1 = min(p[2] - p[0].shape[0] * res for p in parts)
        W, H = int(round((gx1 - gx0) / res)), int(round((gy0 - gy1) / res))
        m = np.full((H, W), np.nan)
        for a, x0, y0, _ in parts:
            c0, r0 = int(round((x0 - gx0) / res)), int(round((gy0 - y0) / res))
            sub = m[r0:r0 + a.shape[0], c0:c0 + a.shape[1]]
            empty = ~np.isfinite(sub)
            sub[empty] = a[empty]
        yield m, gx0, gy0, res


def sample_rasters(paths: list, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Values of a mosaic of rasters at points (cell centres of our grid:
    xs [W] eastings, ys [H] northings) -> [H, W], NaN where no raster has
    data. A raster's value is taken to sit at its own pixel centre and is
    interpolated bilinearly; points that need a no-data neighbour are NaN
    (so voids do not bleed). When grids coincide this returns the raster
    values exactly. Earlier paths win where rasters overlap."""
    from scipy.ndimage import map_coordinates

    out = np.full((ys.size, xs.size), np.nan)
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
    for a, x0, y0, res in _mosaic_windows(paths, xmin, ymin, xmax, ymax):
        fc = (xs - x0) / res - 0.5
        fr = (y0 - ys) / res - 0.5
        inside_c = (fc >= -1e-6) & (fc <= a.shape[1] - 1 + 1e-6)
        inside_r = (fr >= -1e-6) & (fr <= a.shape[0] - 1 + 1e-6)
        if not inside_c.any() or not inside_r.any():
            continue
        R, C = np.meshgrid(np.clip(fr, 0, a.shape[0] - 1), np.clip(fc, 0, a.shape[1] - 1), indexing="ij")
        ok = np.isfinite(a)
        v = map_coordinates(np.where(ok, a, 0.0), [R, C], order=1, mode="nearest", prefilter=False)
        w = map_coordinates(ok.astype(np.float64), [R, C], order=1, mode="nearest", prefilter=False)
        good = (w > 1 - 1e-6) & inside_r[:, None] & inside_c[None, :] & ~np.isfinite(out)
        out[good] = v[good] / w[good]
    return out


def build_vrt(vrt_path: str | Path, tiles: list, width: int, height: int, xmin: float, ymax: float,
              gsd: float, crs_wkt: str | None, bands: int = 1, dtype: str = "Float32",
              nodata: float | None = -9999.0, colorinterp: list | None = None) -> Path:
    """Write a GDAL VRT mosaic. tiles: [(path, row0, col0, h, w)] on the mosaic grid.
    Float tiles use ComplexSource with NODATA so a tile's no-data never hides a
    neighbour's data."""
    from xml.sax.saxutils import escape
    vrt_path = Path(vrt_path)
    out = [f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">']
    if crs_wkt:
        out.append(f"  <SRS>{escape(crs_wkt)}</SRS>")
    out.append(f"  <GeoTransform>{xmin!r}, {gsd!r}, 0.0, {ymax!r}, 0.0, {-gsd!r}</GeoTransform>")
    for b in range(1, bands + 1):
        out.append(f'  <VRTRasterBand dataType="{dtype}" band="{b}">')
        if nodata is not None:
            out.append(f"    <NoDataValue>{nodata}</NoDataValue>")
        if colorinterp:
            out.append(f"    <ColorInterp>{colorinterp[b - 1]}</ColorInterp>")
        for path, r0, c0, h, w in tiles:
            rel = Path(path).resolve().relative_to(vrt_path.parent.resolve())
            tag = "ComplexSource" if nodata is not None else "SimpleSource"
            out.append(f"    <{tag}>")
            out.append(f'      <SourceFilename relativeToVRT="1">{escape(rel.as_posix())}</SourceFilename>')
            out.append(f"      <SourceBand>{b}</SourceBand>")
            out.append(f'      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>')
            out.append(f'      <DstRect xOff="{c0}" yOff="{r0}" xSize="{w}" ySize="{h}"/>')
            if nodata is not None:
                out.append(f"      <NODATA>{nodata}</NODATA>")
            out.append(f"    </{tag}>")
        out.append("  </VRTRasterBand>")
    out.append("</VRTDataset>")
    vrt_path.write_text("\n".join(out))
    return vrt_path


def vrt_to_geotiff(vrt_path: str | Path, out_path: str | Path, rgba: bool = False):
    """Materialise a VRT as one tiled, LZW-compressed GeoTIFF (streams; the
    mosaic is never held in memory)."""
    opts = {"compress": "lzw", "tiled": True, "BIGTIFF": "IF_SAFER"}
    if rgba:
        opts.update({"photometric": "RGB", "alpha": "YES"})
    try:
        import rasterio.shutil
        rasterio.shutil.copy(str(vrt_path), str(out_path), driver="GTiff", **opts)
        return
    except ImportError:
        pass
    from osgeo import gdal
    co = ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"] + (["PHOTOMETRIC=RGB", "ALPHA=YES"] if rgba else [])
    gdal.Translate(str(out_path), str(vrt_path), creationOptions=co)


def epsg_of(crs_wkt: str | None) -> int | None:
    """EPSG code of a WKT CRS, if it can be identified."""
    if not crs_wkt:
        return None
    try:
        from rasterio.crs import CRS
        return CRS.from_wkt(crs_wkt).to_epsg()
    except ImportError:
        pass
    except Exception:
        return None
    try:
        from osgeo import osr
        srs = osr.SpatialReference()
        if srs.ImportFromWkt(crs_wkt) != 0:
            return None
        srs.AutoIdentifyEPSG()
        code = srs.GetAuthorityCode(None)
        return int(code) if code else None
    except Exception:
        return None


def build_overviews(path: str | Path, resampling: str = "average", min_size: int = 256):
    """Internal overviews (x2, x4, ...) so large mosaics draw quickly in QGIS/LP360."""
    try:
        import rasterio
        from rasterio.enums import Resampling
        with rasterio.open(str(path), "r+") as ds:
            factors, f = [], 2
            while max(ds.width, ds.height) / f >= min_size:
                factors.append(f)
                f *= 2
            if factors:
                ds.build_overviews(factors, getattr(Resampling, resampling))
                ds.update_tags(ns="rio_overview", resampling=resampling)
        return
    except ImportError:
        pass
    from osgeo import gdal
    ds = gdal.Open(str(path), gdal.GA_Update)
    factors, f = [], 2
    while max(ds.RasterXSize, ds.RasterYSize) / f >= min_size:
        factors.append(f)
        f *= 2
    if factors:
        ds.BuildOverviews(resampling.upper(), factors)
    ds = None


def iter_rows(path: str | Path, rows: int = 1024):
    """Yield (row0, block [h, W] float64 with NaN for no-data) down a raster."""
    try:
        import rasterio
        from rasterio.windows import Window
        with rasterio.open(str(path)) as src:
            nd = src.nodata
            for r0 in range(0, src.height, rows):
                h = min(rows, src.height - r0)
                a = src.read(1, window=Window(0, r0, src.width, h)).astype(np.float64)
                if nd is not None:
                    a[a == nd] = np.nan
                yield r0, a
        return
    except ImportError:
        pass
    from osgeo import gdal
    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    nd = band.GetNoDataValue()
    for r0 in range(0, ds.RasterYSize, rows):
        h = min(rows, ds.RasterYSize - r0)
        a = band.ReadAsArray(0, r0, ds.RasterXSize, h).astype(np.float64)
        if nd is not None:
            a[a == nd] = np.nan
        yield r0, a

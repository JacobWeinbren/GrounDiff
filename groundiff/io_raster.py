"""GeoTIFF read/write via rasterio, or GDAL's Python bindings (as in QGIS)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


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


def read_geotiff(path: str | Path) -> tuple[np.ndarray, dict]:
    """Returns (array with NaN for no-data, info{xmin, ymax, gsd, crs_wkt})."""
    try:
        import rasterio
        with rasterio.open(str(path)) as src:
            a = src.read(1).astype(np.float64)
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            tr = src.transform
            if abs(tr.a) != abs(tr.e) or tr.b or tr.d:
                raise ValueError(f"{path}: only north-up rasters with square pixels are supported")
            return a, {"xmin": tr.c, "ymax": tr.f, "gsd": tr.a,
                       "crs_wkt": src.crs.to_wkt() if src.crs else None}
    except ImportError:
        pass
    from osgeo import gdal
    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    a = band.ReadAsArray().astype(np.float64)
    nd = band.GetNoDataValue()
    if nd is not None:
        a[a == nd] = np.nan
    gt = ds.GetGeoTransform()
    if gt[2] or gt[4] or abs(gt[1]) != abs(gt[5]):
        raise ValueError(f"{path}: only north-up rasters with square pixels are supported")
    return a, {"xmin": gt[0], "ymax": gt[3], "gsd": gt[1], "crs_wkt": ds.GetProjection() or None}

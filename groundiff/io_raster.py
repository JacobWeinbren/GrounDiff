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
        srs.ImportFromEPSG(epsg)
        return srs.ExportToWkt()
    except ImportError:
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

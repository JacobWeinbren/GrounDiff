"""Pre-rendered colour overlays for viewing over a DTM in LP360 (or any GIS).

LP360 draws rasters as images, so the priority maps are delivered already
coloured, with transparency where there is nothing to look at:

  <name>_overlay.tif      4-band RGBA GeoTIFF (alpha channel), LZW, + .tfw/.prj
  <name>_overlay_rgb.tif  3-band RGB GeoTIFF with nodata = 0 for viewers that
                          ignore alpha (set the layer's transparency in the viewer)
  <name>.qml              QGIS style for the float raster (applied automatically
                          when QGIS opens <name>.tif)

Colour ramp: one hue (magenta -> deep purple, the complement of the light
green used for DTM shading), getting darker AND more opaque with the value, and
fully transparent below the first stop so the DTM shows through where no
attention is needed. The blended colours were checked (lightness monotone,
visible steps, light end >= 2:1 contrast) over pale, mid and hillshade greens,
white and grey backgrounds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

RAMP = [("#B8198F", 0.62), ("#8E1080", 0.75), ("#750C76", 0.84), ("#580967", 0.91), ("#3A0650", 0.96)]

PRESETS = {
    # p_edit: probability that editors change the lasground_new DTM here
    "edit": {"stops": [0.2, 0.4, 0.6, 0.8, 1.0], "label": "Edit probability"},
    # |dz_before|: size of the predicted correction, metres
    "dz": {"stops": [0.15, 0.3, 0.5, 1.0, 2.0], "label": "Predicted edit |dz| (m)", "abs": True},
    # std: spread across samples / TTA views, metres
    "uncertainty": {"stops": [0.1, 0.2, 0.4, 0.7, 1.0], "label": "Uncertainty (m)"},
    # p_ground: invert so LOW confidence is highlighted
    "low_confidence": {"stops": [0.2, 0.4, 0.6, 0.8, 1.0], "label": "1 - confidence", "invert": True},
}


def _rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float64)


def prepare_values(values: np.ndarray, preset: str) -> np.ndarray:
    p = PRESETS[preset]
    v = np.asarray(values, np.float64)
    if p.get("abs"):
        v = np.abs(v)
    if p.get("invert"):
        v = 1.0 - v
    return v


def render_rgba(values: np.ndarray, preset: str = "edit") -> np.ndarray:
    """values [H, W] -> uint8 RGBA [H, W, 4]; transparent below the first stop and at NaN."""
    stops = PRESETS[preset]["stops"]
    v = prepare_values(values, preset)
    cols = np.stack([_rgb(c) for c, _ in RAMP])
    alph = np.array([a for _, a in RAMP])
    out = np.zeros(v.shape + (4,), np.float64)
    ok = np.isfinite(v) & (v >= stops[0])
    vv = np.clip(v[ok], stops[0], stops[-1])
    for ch in range(3):
        out[..., ch][ok] = np.interp(vv, stops, cols[:, ch])
    out[..., 3][ok] = np.interp(vv, stops, alph) * 255.0
    return np.round(out).astype(np.uint8)


def _write_sidecars(path: Path, xmin: float, ymax: float, gsd: float, crs_wkt: str | None):
    # world file: pixel size x, rotation, rotation, -pixel size y, centre of the top-left pixel
    path.with_suffix(".tfw").write_text(
        f"{gsd:.10f}\n0.0\n0.0\n{-gsd:.10f}\n{xmin + gsd / 2:.10f}\n{ymax - gsd / 2:.10f}\n")
    if crs_wkt:
        try:
            from pyproj import CRS
            wkt = CRS.from_wkt(crs_wkt).to_wkt("WKT1_ESRI")
        except Exception:
            wkt = crs_wkt
        path.with_suffix(".prj").write_text(wkt)


def write_rgba_geotiff(path: str | Path, rgba: np.ndarray, xmin: float, ymax: float, gsd: float,
                       crs_wkt: str | None = None, alpha: bool = True):
    """alpha=True: 4-band RGBA. alpha=False: 3-band RGB, transparent pixels = 0 = nodata."""
    path = Path(path)
    bands = rgba if alpha else np.where(rgba[..., 3:4] > 0, np.maximum(rgba[..., :3], 1), 0).astype(np.uint8)
    n = bands.shape[-1]
    try:
        import rasterio
        from rasterio.enums import ColorInterp
        from rasterio.transform import from_origin
        prof = dict(driver="GTiff", height=bands.shape[0], width=bands.shape[1], count=n, dtype="uint8",
                    crs=crs_wkt, transform=from_origin(xmin, ymax, gsd, gsd), compress="lzw",
                    tiled=True, photometric="RGB")
        if alpha:
            prof["alpha"] = "YES"                  # GDAL: extra sample is unassociated alpha
        else:
            prof["nodata"] = 0
        with rasterio.open(str(path), "w", **prof) as dst:
            dst.write(np.moveaxis(bands, -1, 0))
            dst.colorinterp = ([ColorInterp.red, ColorInterp.green, ColorInterp.blue]
                               + ([ColorInterp.alpha] if alpha else []))
    except ImportError:
        from osgeo import gdal, osr
        opts = ["COMPRESS=LZW", "TILED=YES", "PHOTOMETRIC=RGB"] + (["ALPHA=YES"] if alpha else [])
        ds = gdal.GetDriverByName("GTiff").Create(str(path), bands.shape[1], bands.shape[0], n, gdal.GDT_Byte, opts)
        ds.SetGeoTransform((xmin, gsd, 0.0, ymax, 0.0, -gsd))
        if crs_wkt:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(crs_wkt)
            ds.SetProjection(srs.ExportToWkt())
        for i in range(n):
            b = ds.GetRasterBand(i + 1)
            b.WriteArray(bands[..., i])
            if not alpha:
                b.SetNoDataValue(0)
        ds.FlushCache()
    _write_sidecars(path, xmin, ymax, gsd, crs_wkt)


def qml_style(preset: str) -> str:
    """QGIS singleband-pseudocolour style with per-entry opacity (QGIS >= 3.18
    honours colour alpha in pseudocolour ramps)."""
    p = PRESETS[preset]
    stops = p["stops"]
    items = []
    # below the first stop: transparent
    first = _rgb(RAMP[0][0]).astype(int)
    items.append(f'<item alpha="0" value="{stops[0] - 1e-6}" label="&lt; {stops[0]}" '
                 f'color="#{first[0]:02x}{first[1]:02x}{first[2]:02x}"/>')
    for s, (c, a) in zip(stops, RAMP):
        items.append(f'<item alpha="{int(round(a * 255))}" value="{s}" label="{s}" color="{c.lower()}"/>')
    if p.get("invert") or p.get("abs"):
        note = "<!-- note: this preset transforms values (invert/abs); the style applies to raw values -->"
    else:
        note = ""
    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.22" styleCategories="Symbology">{note}
 <pipe>
  <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" alphaBand="-1"
                  classificationMin="{stops[0]}" classificationMax="{stops[-1]}">
   <rastershader>
    <colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0">
     {chr(10).join('     ' + i for i in items)}
    </colorrampshader>
   </rastershader>
  </rasterrenderer>
 </pipe>
</qgis>
"""


def write_overlays(values: np.ndarray, out_stem: str | Path, preset: str, xmin: float, ymax: float,
                   gsd: float, crs_wkt: str | None = None) -> list[Path]:
    """Write <stem>_overlay.tif (RGBA), <stem>_overlay_rgb.tif (RGB+nodata) and <stem>.qml."""
    stem = Path(out_stem)
    rgba = render_rgba(values, preset)
    a = stem.with_name(stem.name + "_overlay.tif")
    b = stem.with_name(stem.name + "_overlay_rgb.tif")
    write_rgba_geotiff(a, rgba, xmin, ymax, gsd, crs_wkt, alpha=True)
    write_rgba_geotiff(b, rgba, xmin, ymax, gsd, crs_wkt, alpha=False)
    paths = [a, b]
    if not (PRESETS[preset].get("invert") or PRESETS[preset].get("abs")):
        q = stem.with_suffix(".qml")
        q.write_text(qml_style(preset))
        paths.append(q)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raster", help="float GeoTIFF, e.g. p_edit.tif")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="edit")
    ap.add_argument("--out", help="output stem (default: next to the input)")
    a = ap.parse_args(argv)
    from .io_raster import read_geotiff
    arr, info = read_geotiff(a.raster)
    stem = Path(a.out) if a.out else Path(a.raster).with_suffix("")
    for p in write_overlays(arr, stem, a.preset, info["xmin"], info["ymax"], info["gsd"], info["crs_wkt"]):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

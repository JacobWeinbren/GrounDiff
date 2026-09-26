import xml.etree.ElementTree as ET

import numpy as np

from groundiff.overlay import render_rgba, write_overlays


def test_render_transparent_below_threshold_and_monotone():
    v = np.array([[np.nan, 0.0, 0.19, 0.2, 0.5, 1.0, 1.5]])
    rgba = render_rgba(v, "edit")
    assert (rgba[0, :3, 3] == 0).all()                       # NaN and below 0.2: transparent
    a = rgba[0, 3:, 3].astype(int)
    lum = rgba[0, 3:, :3].astype(float) @ [0.2126, 0.7152, 0.0722]
    assert (np.diff(a) >= 0).all() and (np.diff(lum) <= 0).all()   # more opaque and darker with value
    assert rgba[0, 5, 3] == rgba[0, 6, 3]                     # clipped above the last stop


def test_presets_transform():
    assert render_rgba(np.array([[-0.8]]), "dz")[0, 0, 3] > 0             # |dz| used
    assert render_rgba(np.array([[0.1]]), "nonground")[0, 0, 3] > 0  # 1 - 0.1 = 0.9


def test_write_overlays(tmp_path):
    import rasterio
    v = np.linspace(0, 1, 64 * 32).reshape(32, 64)
    paths = write_overlays(v, tmp_path / "p_edit", "edit", 400000.0, 200032.0, 0.5, None)
    names = {p.name for p in paths}
    assert names == {"p_edit_overlay.tif", "p_edit_overlay_rgb.tif", "p_edit.qml"}
    with rasterio.open(tmp_path / "p_edit_overlay.tif") as src:
        assert src.count == 4 and src.colorinterp[3].name == "alpha"
        assert src.transform.c == 400000.0 and src.transform.a == 0.5
        data = src.read()
    assert data[3, 0, 0] == 0 and data[3, -1, -1] > 200
    with rasterio.open(tmp_path / "p_edit_overlay_rgb.tif") as src:
        assert src.count == 3 and src.nodata == 0
        rgb = src.read()
    assert (rgb[:, 0, 0] == 0).all() and (rgb[:, -1, -1] > 0).all()
    tfw = (tmp_path / "p_edit_overlay.tfw").read_text().split()
    assert float(tfw[0]) == 0.5 and float(tfw[4]) == 400000.25
    ET.fromstring((tmp_path / "p_edit.qml").read_text().split("\n", 1)[1])   # valid XML after doctype

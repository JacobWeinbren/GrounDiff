"""EA DTM downloader, offline (the Defra service is mocked)."""
import io
import json
import zipfile

import numpy as np

from groundiff.data import ea_dtm
from groundiff.io_raster import write_geotiff

NAMES = ["TL4378nw_P_12534_20220315_20220316.copc.laz", "TL4378ne_P_12534_20220315_20220316.copc.laz",
         "TL4379sw_P_12535_20220127_20220130.copc.laz", "TL4379se_P_12534_20220315_20220316.copc.laz"]


def test_tile_maths():
    t = ea_dtm.tile5k(543250, 278750)
    assert t["id"] == "TL4075" and t["label"] == "TL47nw" and t["bounds"] == (540000, 275000, 545000, 280000)
    assert ea_dtm.tile5k(441000, 171000)["label"] == "SU47sw"
    want = ea_dtm.wanted_tiles(NAMES)
    assert list(want) == [("TL4075", "2022")]
    assert len(ea_dtm.tiles_for((544900, 279900, 545100, 280100))) == 4          # straddles a corner
    assert ea_dtm.survey_year("SU6570ne_P_11111_20190704_20190705.laz") == "2019"


def test_polygon_is_lonlat_inside_tile():
    poly = ea_dtm.polygon((540000, 275000, 545000, 280000))
    lon, lat = zip(*poly["coordinates"][0])
    assert 0.0 < min(lon) < max(lon) < 0.2 and 52.3 < min(lat) < max(lat) < 52.45


def test_run_downloads_extracts_and_resumes(tmp_path, monkeypatch):
    tif = tmp_path / "src.tif"
    write_geotiff(tif, np.full((10, 10), 12.5), 540000.0, 280000.0, 500.0, nodata=-3.4028235e38)
    calls = {"search": 0, "fetch": 0, "expect": "/lidar_tiles_dtm/2022/0.5/TL4075?subscription-key=dspui"}

    def fake_search(bounds, **kw):
        calls["search"] += 1
        return [{"product": "national_lidar_programme_dtm", "year": "2022", "res": "1", "tile": "TL4075",
                 "label": "TL47nw"},
                {"product": "lidar_tiles_dtm", "year": "2022", "res": "2", "tile": "TL4075", "label": "TL47nw"},
                {"product": "lidar_tiles_dtm", "year": "2022", "res": "0.5", "tile": "TL4075", "label": "TL47nw"},
                {"product": "lidar_tiles_dtm", "year": "2021", "res": "1", "tile": "TL4075", "label": "TL47nw"},
                {"product": "lidar_composite_dtm", "year": "2022", "res": "1", "tile": "TL4075", "label": "TL47nw"}]

    def fake_fetch(url, dst, **kw):
        calls["fetch"] += 1
        assert url.endswith(calls["expect"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.write(tif, "TL47nw_DTM_1m.tif")
            z.writestr("TL47nw_DTM_1m_Metadata.gpkg", b"x")
        dst.write_bytes(buf.getvalue())
        return dst

    monkeypatch.setattr(ea_dtm, "search", fake_search)
    monkeypatch.setattr(ea_dtm, "fetch_zip", fake_fetch)
    pts = [tmp_path / n for n in NAMES]
    r = ea_dtm.run(pts, tmp_path / "dtm", log=lambda *a: None)
    assert not r["failed"] and calls["search"] == 1 and calls["fetch"] == 1
    rec = r["manifest"]["TL4075/2022/auto"]
    assert rec["product"] == "lidar_tiles_dtm" and rec["res"] == "0.5"          # survey's own DTM; 1 m, else 50 cm
    assert rec["files"][0].endswith("TL47nw_DTM_1m.tif")
    assert json.loads((tmp_path / "dtm" / "manifest.json").read_text())
    r2 = ea_dtm.run(pts, tmp_path / "dtm", log=lambda *a: None)                  # resumes: nothing to do
    assert calls["fetch"] == 1 and not r2["failed"]
    # forcing the NLP product; and a year the service does not offer is reported, not downloaded
    calls["expect"] = "/national_lidar_programme_dtm/2022/1/TL4075?subscription-key=dspui"
    r3 = ea_dtm.run(pts, tmp_path / "dtm3", product="national_lidar_programme_dtm", log=lambda *a: None)
    assert not r3["failed"] and calls["fetch"] == 2
    r4 = ea_dtm.run(pts, tmp_path / "dtm2", year="2020", log=lambda *a: None)
    assert r4["failed"] and calls["fetch"] == 2

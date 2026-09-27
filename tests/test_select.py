"""Stratified tile choice: coordinates, OpenStreetMap parsing, strata, quotas, bridge sampling."""
import json
from collections import Counter

import numpy as np
import pytest

from groundiff.data import select as S


def test_bng_matches_ordnance_survey():
    # OS worked example (projection only, OSGB36 latitude/longitude)
    e, n = S._tm(np.radians(52 + 39 / 60 + 27.2531 / 3600), np.radians(1 + 43 / 60 + 4.5177 / 3600))
    assert abs(e - 651409.903) < 0.01 and abs(n - 313177.270) < 0.01
    e, n = S.wgs84_to_bng(51.500729, -0.124625)                       # Big Ben, ~(530268, 179640)
    assert abs(e - 530268) < 10 and abs(n - 179640) < 10
    lat, lon = S.bng_to_wgs84(e, n)
    assert abs(lat - 51.500729) < 1e-6 and abs(lon + 0.124625) < 1e-6


def test_overpass_parsing_and_counting():
    lat, lon = S.bng_to_wgs84(401000, 201000)
    geom = [{"lat": lat, "lon": lon}, {"lat": lat + 0.0005, "lon": lon}]
    data = {"elements": [
        {"type": "way", "tags": {"highway": "primary", "bridge": "yes"}, "geometry": geom},
        {"type": "way", "tags": {"railway": "rail", "bridge": "viaduct"}, "geometry": geom},
        {"type": "way", "tags": {"highway": "primary", "bridge": "no"}, "geometry": geom},
        {"type": "way", "tags": {"man_made": "dyke"}, "geometry": geom},
        {"type": "way", "tags": {"wall": "flood_wall"}, "geometry": geom},
        {"type": "way", "tags": {"landuse": "quarry"}, "geometry": geom},
        {"type": "relation", "tags": {"natural": "wetland"}, "center": {"lat": lat, "lon": lon}},
        {"type": "way", "tags": {"building": "yes"}, "geometry": geom},
    ]}
    f = S.parse_overpass(data)
    assert {k: len(v) for k, v in f.items()} == {"bridge": 2, "flood_defence": 2, "quarry": 1, "marsh": 1}
    e0, n0 = f["bridge"][0][0]
    assert abs(e0 - 401000) < 0.5 and abs(n0 - 201000) < 0.5
    tiles = {"a": {"bounds": (400000, 200000, 402000, 202000)}, "b": {"bounds": (402000, 200000, 404000, 202000)}}
    c = S.count_in_tiles(f, tiles)
    assert c["a"] == Counter({"bridge": 2, "flood_defence": 2, "quarry": 1, "marsh": 1}) and not c.get("b")


def _t(**kw):
    t = {"coverage": 1.0, "z_ground_p1": 50, "building": 0.0, "high_veg": 0.1, "relief": 20, "z_ground_p50": 60}
    t.update(kw)
    return t


def test_strata():
    assert S.stratum(_t(building=0.3), Counter(quarry=1)) == "quarry"
    assert S.stratum(_t(), Counter(flood_defence=2)) == "flood_defence"
    assert S.stratum(_t(), Counter(marsh=1)) == "marsh"
    assert S.stratum(_t(coverage=0.6, z_ground_p1=2), Counter()) == "coastal"
    assert S.stratum(_t(building=0.2), Counter()) == "urban"
    assert S.stratum(_t(building=0.07), Counter()) == "suburban"
    assert S.stratum(_t(building=0.02), Counter()) == "village"
    assert S.stratum(_t(high_veg=0.5), Counter()) == "woodland"
    assert S.stratum(_t(relief=300), Counter()) == "upland"
    assert S.stratum(_t(), Counter(bridge=3)) == "farmland"


def test_choose_fills_quotas_and_spreads():
    rng = np.random.default_rng(0)
    tiles = {}
    for i in range(2000):
        s = "farmland" if i % 10 else ["quarry", "marsh", "urban", "upland", "woodland"][i // 10 % 5]
        tiles[f"k{i}"] = {"grid": ["SU", "TQ", "SX", "NZ"][i % 4], "stratum": s,
                          "osm": {"bridge": int(rng.integers(1, 4))} if i % 7 == 0 else {}}
    got = S.choose(tiles, 200)
    assert len(got) == len(set(got)) == 200
    c = Counter(tiles[k]["stratum"] for k in got)
    for s in ("quarry", "marsh", "urban", "upland", "woodland"):
        assert c[s] >= round(S.QUOTAS[s] * 200)
    assert sum(1 for k in got if tiles[k]["osm"].get("bridge")) >= round(S.BRIDGE_QUOTA * 200)
    g = Counter(tiles[k]["grid"] for k in got)
    assert max(g.values()) - min(g.values()) <= 2


def test_line_mask():
    from groundiff.data.rasterise import Grid, line_mask
    g = Grid(0.0, 100.0, 1.0, 100, 100)
    m = line_mask(g, [[[10.0, 50.0], [90.0, 50.0]], [[500.0, 500.0], [600.0, 600.0]]], 5.0)
    assert m[50, 50] and m[45, 50] and not m[40, 50] and not m[50, 2]
    assert not line_mask(g, [[[500.0, 500.0], [600.0, 600.0]]], 5.0).any()


def test_bridge_oversampling(tmp_path):
    """Tiles centred on bridge cells make up about bridge_oversample of the training tiles."""
    import torch
    from groundiff.data.dataset import DataConfig, TileDataset
    sd = tmp_path / "s1"
    sd.mkdir()
    H = W = 400
    for n in ("dsm_max", "dsm_min", "gt_dtm"):
        np.save(sd / f"{n}.npy", np.full((H, W), 5.0, np.float32))
    np.save(sd / "gt_valid.npy", np.ones((H, W), np.uint8))
    br = np.zeros((H, W), np.uint8)
    br[300:305, 20:40] = 1
    np.save(sd / "bridge.npy", br)
    (sd / "meta.json").write_text(json.dumps({"grid": {"height": H, "width": W, "gsd": 1.0, "xmin": 0, "ymax": 0},
                                              "bridge_cells": int(br.sum())}))
    cfg = DataConfig(root=str(tmp_path), tile=64, samples_per_epoch=200, bridge_oversample=0.5, augment=False,
                     cond_channels=["dsm_max", "dsm_min"])
    ds = TileDataset(cfg, split=None, mode="train")
    assert len(ds.bridges) == 1
    seen = []
    real = ds._exact
    ds._exact = lambda sc, r0, c0, k: seen.append((r0, c0)) or real(sc, r0, c0, k)
    rng = np.random.default_rng(1)
    for _ in range(400):
        ds._sample_train(rng)
    on = [r0 <= 302 < r0 + 64 and c0 <= 30 < c0 + 64 for r0, c0 in seen]
    assert len(seen) == 400 and 0.4 < np.mean(on) < 0.75        # ~50 % centred + a few random hits
    assert torch.isfinite(ds[0]["target"]).all()


def test_select_main_end_to_end(tmp_path, monkeypatch):
    """The CLI with the bucket listing, the lidar scan and Overpass faked (no network)."""
    from groundiff.data.osgrid import parse_tile
    keys = [f"data/UK/DEFRA/LIDAR_2022/copc/SU{e:02d}{n:02d}_P_1_20220101_20220101.copc.laz"
            for e in range(0, 20, 2) for n in range(0, 20, 2)]
    monkeypatch.setattr(S, "list_keys", lambda: [(k, 10_000_000) for k in keys])

    def scan(key, levels=2):
        x0, y0, x1, y1 = parse_tile(key)["extent"]
        e = parse_tile(key)["origin"][0]
        return {"key": key, "bounds": (x0, y0, x1, y1), "points": 1, "sampled": 1, "ground": 0.5,
                "building": 0.2 if e % 8000 == 0 else 0.0, "high_veg": 0.1, "z_ground_p1": 50,
                "z_ground_p50": 60, "relief": 10, "coverage": 1.0}
    monkeypatch.setattr(S, "scan_tile", scan)
    x0, y0 = parse_tile(keys[0])["origin"]
    lat, lon = S.bng_to_wgs84(x0 + 500, y0 + 500)
    reply = {"elements": [{"type": "way", "tags": {"highway": "a", "bridge": "yes"},
                           "geometry": [{"lat": lat, "lon": lon}, {"lat": lat + 1e-4, "lon": lon}]}]}
    queries = []

    class R:
        def __init__(self, req, timeout=None):
            queries.append(req.data)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(reply).encode()
    monkeypatch.setattr(S.urllib.request, "urlopen", R)
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    assert S.main(["--out", str(tmp_path), "--target", "20", "--workers", "2"]) == 0
    sel = json.loads((tmp_path / "selection.json").read_text())["tiles"]
    assert len(sel) == 20 and sel[0]["key"] == keys[0] and sel[0]["bridges"] == 1
    assert json.loads((tmp_path / "osm.json").read_text())["bridge"]
    assert len(queries) == 1
    # rerun: the scan and OSM come from the cache
    monkeypatch.setattr(S, "scan_tile", lambda *a: pytest.fail("rescanned"))
    assert S.main(["--out", str(tmp_path), "--target", "20"]) == 0 and len(queries) == 1


def test_stratum_table():
    from groundiff.infer import stratum_table
    rows = [{"stratum": "urban", "sq_err_sum": 4.0, "n_valid": 100, "bridge": {"rmse": 0.5, "cells": 10}},
            {"stratum": "farmland", "sq_err_sum": 1.0, "n_valid": 100}]
    t = stratum_table(rows)
    assert "urban" in t and "near bridges" in t and "0.500" in t and "0.158" in t   # all: sqrt(5 / 200)

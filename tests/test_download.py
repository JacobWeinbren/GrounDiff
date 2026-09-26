from groundiff.data.download import floor_fill, neighbours, os_origin, parse_tile


def test_os_grid_origins():
    assert os_origin("SU", 0, 0) == (400000, 100000)
    assert os_origin("NU", 0, 0) == (400000, 600000)
    assert os_origin("TQ", 30, 80) == (530000, 180000)          # central London
    t = parse_tile("data/UK/DEFRA/LIDAR_2022/copc/NU0445ne_P_12498_20220119_20220119.copc.laz")
    assert t["grid"] == "NU" and t["square"] == "NU0445" and t["quad"] == "ne"
    assert t["origin"] == (404500, 645500)


def test_floor_fill_and_neighbours():
    keys = [f"SU{e:02d}{n:02d}{q}_x.copc.laz" for e in range(10) for n in range(10) for q in ("ne", "nw", "se", "sw")]
    keys += [f"TQ{e:02d}00ne_x.copc.laz" for e in range(5)]
    chosen = floor_fill(keys, target=20, min_per_grid=3, seed=0)
    assert len(chosen) == 20 and sum(k.startswith("TQ") for k in chosen) >= 3
    block = neighbours(["SU0505ne_x.copc.laz"], keys, size=3)
    assert len(block) == 9 and "SU0606sw_x.copc.laz" in block and "SU0505sw_x.copc.laz" in block

"""Targets from published DTM rasters, the quality gate, and the EA class rules."""
import json

import numpy as np
import pytest

from groundiff.data.download import neighbours
from groundiff.data.osgrid import parse_tile
from groundiff.data.preprocess import check_lasground_classes, index_rasters, process_scene, rasters_for
from groundiff.data.rasterise import Grid
from groundiff.data.split import block_split
from groundiff.io_raster import sample_rasters, write_geotiff
from tests.synthetic import terrain, write_scene

X0, Y0 = 400000.0, 200000.0


def plane(x, y):
    return 10.0 + 0.3 * (x - X0) - 0.1 * (y - Y0)


def raster_of(fn, xmin, ymax, w, h, res):
    xs = xmin + (np.arange(w) + 0.5) * res
    ys = ymax - (np.arange(h) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    return fn(X, Y)


def test_sample_rasters_exact_when_aligned(tmp_path):
    a = raster_of(plane, X0, Y0 + 50, 50, 50, 1.0)
    a[10:13, 20:24] = np.nan
    write_geotiff(tmp_path / "t.tif", a, X0, Y0 + 50, 1.0)
    g = Grid.from_bounds(X0 + 5, Y0 + 5, X0 + 45, Y0 + 45, 1.0)
    xs, ys = g.cell_centres()
    v = sample_rasters([str(tmp_path / "t.tif")], xs, ys)
    truth = raster_of(plane, g.xmin, g.ymax, g.width, g.height, 1.0)
    ok = np.isfinite(v)
    assert np.abs(v - truth)[ok].max() < 1e-4
    assert (~ok).sum() == 12                        # the hole, and nothing around it


def test_sample_rasters_resamples_and_mosaics(tmp_path):
    # two 0.5 m rasters side by side; a 1 m grid across the seam (cell centres fall between pixels)
    for i, x0 in enumerate((X0, X0 + 20)):
        write_geotiff(tmp_path / f"r{i}.tif", raster_of(plane, x0, Y0 + 20, 40, 40, 0.5), x0, Y0 + 20, 0.5)
    g = Grid.from_bounds(X0 + 2, Y0 + 2, X0 + 38, Y0 + 18, 1.0)
    xs, ys = g.cell_centres()
    v = sample_rasters([str(tmp_path / "r0.tif"), str(tmp_path / "r1.tif")], xs, ys)
    truth = raster_of(plane, g.xmin, g.ymax, g.width, g.height, 1.0)
    assert np.isfinite(v).all() and np.abs(v - truth).max() < 1e-4


@pytest.fixture(scope="module")
def laz(tmp_path_factory):
    root = tmp_path_factory.mktemp("tg")
    after, before = write_scene(root / "laz", name="SX0000_dtm", size=96.0)
    return root, after, before


def _dtm(root, offset=0.0, name="dtm", noise=0.0, analytic=False):
    """A stand-in EA DTM built as the EA does: TIN of the survey's (edited)
    ground points at 1 m cell centres. noise > 0 imitates a DTM from another
    survey (independent measurement noise); analytic=True uses the terrain
    function instead of the points."""
    from groundiff.data.laz import read_points
    from groundiff.data.rasterise import tin_dtm
    d = root / name
    d.mkdir(exist_ok=True)
    g = Grid(X0, Y0 + 100, 1.0, 100, 100)
    if analytic:
        z = raster_of(lambda X, Y: terrain(X - X0, Y - Y0), X0, Y0 + 100, 100, 100, 1.0)
    else:
        pts = read_points(root / "laz" / "after" / "SX0000_dtm.las")
        k = pts.cls == 2
        z, _ = tin_dtm(g, pts.x[k], pts.y[k], pts.z[k])
    z = z + offset + (np.random.default_rng(1).normal(0, noise, z.shape) if noise else 0.0)
    write_geotiff(d / "SX00sw.tif", np.round(z, 3), X0, Y0 + 100, 1.0)          # EA values are on whole mm
    return d


def test_process_scene_with_dtm_target(laz):
    root, after, before = laz
    d = _dtm(root)
    idx = index_rasters(d)
    hits = rasters_for((X0, Y0, X0 + 96, Y0 + 96), idx)
    meta = process_scene(before, root / "scenes", dtm_paths=hits, gsd=1.0)
    assert meta["target"] == "dtm_raster" and not meta["quality"]["suspect"]
    assert meta["quality"]["agree_frac"] > 0.8
    assert meta["quality"]["exact_frac"] > 0.8                           # same survey: matches the ground TIN
    sd = root / "scenes" / "SX0000_dtm"
    gt, ok = np.load(sd / "gt_dtm.npy"), np.load(sd / "gt_valid.npy") > 0.5
    g = meta["grid"]
    from groundiff.io_raster import read_geotiff
    src, info = read_geotiff(d / "SX00sw.tif")
    c0, r0 = int(round(g["xmin"] - info["xmin"])), int(round(info["ymax"] - g["ymax"]))
    ref = src[r0:r0 + g["height"], c0:c0 + g["width"]]
    assert ok.mean() > 0.9 and np.abs(gt - ref)[ok].max() < 1e-4        # raster values copied exactly
    assert not (sd / "top_ground.npy").exists()
    # cached on the second call; re-made when the target changes
    assert process_scene(before, root / "scenes", dtm_paths=hits, gsd=1.0) is None


def test_quality_gate_flags_other_survey(laz):
    """A DTM from another survey agrees within 0.2 m almost everywhere, but
    not within 5 mm on open ground."""
    root, _, before = laz
    d = _dtm(root, noise=0.03, name="dtm_other")
    meta = process_scene(before, root / "scenes_other", dtm_paths=rasters_for((X0, Y0, X0 + 96, Y0 + 96),
                                                                                index_rasters(d)), gsd=1.0)
    q = meta["quality"]
    assert q["agree_frac"] > 0.9 and q["exact_frac"] < 0.15 and q["suspect"]


def test_flat_water_is_not_a_target():
    from groundiff.data.rasterise import flat_areas
    rng = np.random.default_rng(0)
    z = np.round(50 + np.cumsum(rng.normal(0, 0.01, (60, 60)), 1), 3)   # mm-rounded natural surface
    z[10:30, 20:45] = 2.48                                              # flattened river
    m = flat_areas(z)
    assert m[10:30, 20:45].all() and m.sum() == 20 * 25


def test_quality_gate_flags_mismatched_target(laz):
    root, _, before = laz
    d = _dtm(root, offset=0.5, name="dtm_off")
    meta = process_scene(before, root / "scenes_off", dtm_paths=rasters_for((X0, Y0, X0 + 96, Y0 + 96),
                                                                              index_rasters(d)), gsd=1.0)
    assert meta["quality"]["suspect"] and meta["quality"]["median_dz"] == pytest.approx(0.5, abs=0.05)
    from groundiff.data.dataset import DataConfig, load_scenes
    with pytest.raises(FileNotFoundError):                                # skipped by training
        load_scenes(DataConfig(root=str(root / "scenes_off")), None)
    assert load_scenes(DataConfig(root=str(root / "scenes_off"), include_suspect=True), None)


def test_published_classes_are_rejected_as_before(laz):
    root, after, _ = laz
    with pytest.raises(ValueError, match="lasground_new"):
        process_scene(after, root / "scenes_bad", dtm_paths=[str(_dtm(root) / "SX00sw.tif")], gsd=1.0)
    assert check_lasground_classes({1: 100, 2: 900}) is None
    assert check_lasground_classes({1: 100, 2: 900, 7: 3}) is None      # low noise is fine
    assert "classes [3, 5, 6]" in check_lasground_classes({1: 5, 2: 50, 3: 20, 5: 20, 6: 5})


def test_block_split_balances_uneven_blocks():
    def centres(sizes):
        return {f"b{b}_{i}": (b * 20_000.0 + 5.0, 5.0) for b, n in enumerate(sizes) for i in range(n)}
    for seed in range(5):
        sp = block_split(centres([90, 5, 5]), seed=seed)
        assert len(sp["train"]) == 90 and len(sp["val"]) == 5 and len(sp["test"]) == 5
        sp = block_split(centres([40, 30, 20, 5, 5]), seed=seed)
        assert len(sp["train"]) >= 70 and sp["val"] and sp["test"]
    with pytest.raises(ValueError):
        block_split(centres([50, 50]))


def test_osgrid_and_neighbours():
    t = parse_tile("TL4378nw_P_12534_20220315_20220316.copc.laz")
    assert t["extent"] == (543000, 278500, 543500, 279000) and t["size"] == 500
    k = parse_tile("SU6571.laz")
    assert k["size"] == 1000 and not k["size_known"]
    k2 = parse_tile("NX9410_P_12706_20220902_20220902.copc.laz")      # EA archive 2 km tile
    assert k2["size"] == 2000 and k2["extent"] == (294000, 510000, 296000, 512000)
    keys = ["TL4378nw_P_1_a.laz", "TL4378nw_P_2_b.laz", "TL4378ne_P_1_a.laz", "SU6571_x.laz", "SU6671_y.laz",
            "SU6572_z.laz", "NX9410_a.laz", "NX9610_b.laz"]
    got = neighbours(["TL4378nw_P_1_a.laz", "SU6571_x.laz", "NX9410_a.laz"], keys, 3)
    assert set(got) == set(keys)                                  # both surveys, 1 km and 2 km neighbours


def test_points_only_scene_needs_no_lasground(laz):
    """--points-dir: the published file as downloaded (classes ignored), DTM target."""
    root, after, _ = laz
    d = _dtm(root, name="dtm_p")
    meta = process_scene(after, root / "scenes_pts", dtm_paths=rasters_for((X0, Y0, X0 + 96, Y0 + 96),
                                                                             index_rasters(d)),
                         gsd=1.0, lasground=False)
    sd = root / "scenes_pts" / "SX0000_dtm"
    assert not (sd / "dtm_before.npy").exists() and (sd / "dsm_min.npy").exists()
    q = meta["quality"]
    assert not q["suspect"] and q["agree_frac"] > 0.8 and q["n_ground_cells"] > 50
    d2 = _dtm(root, offset=0.5, name="dtm_p_off")
    meta2 = process_scene(after, root / "scenes_pts2", dtm_paths=rasters_for((X0, Y0, X0 + 96, Y0 + 96),
                                                                               index_rasters(d2)),
                          gsd=1.0, lasground=False)
    assert meta2["quality"]["suspect"]

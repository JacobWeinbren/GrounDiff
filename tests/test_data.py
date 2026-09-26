import json

import numpy as np
import pytest
import torch

from groundiff.data.dataset import DataConfig, Scene, TileDataset
from groundiff.data.preprocess import process_scene
from groundiff.data.rasterise import Grid, class_mode_onehot, rasterise_points, tin_dtm
from groundiff.data.split import block_split
from tests.synthetic import terrain, write_scene


@pytest.fixture(scope="module")
def scene_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("scenes")
    after, before = write_scene(root / "laz", name="SX0000_test", size=128.0)
    meta = process_scene(before, root / "scenes", after=after, gsd=1.0)
    assert meta is not None
    return root / "scenes" / "SX0000_test"


def test_grid_and_rasterise_basic():
    g = Grid.from_bounds(0.2, 0.3, 3.9, 2.1, 1.0)
    assert (g.width, g.height, g.xmin, g.ymax) == (4, 3, 0.0, 3.0)
    x = np.array([0.5, 0.6, 3.5]); y = np.array([2.5, 2.4, 0.5]); z = np.array([1.0, 3.0, 7.0])
    r = rasterise_points(g, x, y, z, np.array([1, 2, 1]), np.array([2, 2, 1]))
    assert r["dsm_max"][0, 0] == 3.0 and r["dsm_min"][0, 0] == 1.0
    assert r["dsm_last"][0, 0] == 3.0            # only the 2nd of 2 returns is "last"
    assert r["density"][0, 0] == 2.0 and r["echoes"][0, 0] == 2.0
    assert r["dsm_max"][2, 3] == 7.0 and np.isnan(r["dsm_max"][1, 1])
    sem = class_mode_onehot(g, x, y, np.array([True, False, False]))
    assert sem[:, 0, 0].tolist() == [1.0, 0.0]   # tie -> ground
    assert sem[:, 2, 3].tolist() == [0.0, 1.0] and sem[:, 1, 1].tolist() == [0.0, 0.0]


def test_tin_recovers_plane():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 50, 4000), rng.uniform(0, 50, 4000)
    z = 3.0 + 0.1 * x - 0.2 * y
    g = Grid.from_bounds(0, 0, 50, 50, 1.0)
    dtm, valid = tin_dtm(g, x + 4e5, y + 2e5, z) if False else tin_dtm(g, x, y, z)
    xs, ys = g.cell_centres()
    X, Y = np.meshgrid(xs, ys)
    truth = 3.0 + 0.1 * X - 0.2 * Y
    assert valid[5:-5, 5:-5].all()
    assert np.nanmax(np.abs(dtm - truth)[valid]) < 1e-4


def test_preprocess_scene_contents(scene_dir):
    meta = json.loads((scene_dir / "meta.json").read_text())
    assert meta["target"] == "after_points" and "quality" in meta
    assert set(meta["class_hist_before"]) <= {"1", "2"}   # lasground_new-style before file
    sc = Scene(scene_dir)
    xs, ys = (np.arange(sc.width) + 0.5) + meta["grid"]["xmin"], meta["grid"]["ymax"] - (np.arange(sc.height) + 0.5)
    X, Y = np.meshgrid(xs, ys)
    lx, ly = X - 400000.0, Y - 200000.0
    truth = terrain(lx, ly)
    gt, ok = sc.array("gt_dtm"), sc.array("gt_valid") > 0.5
    inner = ok & (lx > 2) & (lx < 126) & (ly > 2) & (ly < 126)
    assert np.nanmean(np.abs(gt - truth)[inner]) < 0.05
    dmax = sc.array("dsm_max")
    # every point is kept in the inputs, as in production lasground_new output: noise shows up
    assert np.nanmax(dmax - truth) > 50.0                  # high noise (+60 m)
    assert np.nanmin(sc.array("dsm_min") - truth) < -10.0  # low noise (-15 m)
    roof = (lx > 22) & (lx < 38) & (ly > 22) & (ly < 38)
    assert np.nanmedian((dmax - truth)[roof]) > 7.5
    trees = (lx > 82) & (lx < 108) & (ly > 72) & (ly < 108)
    assert np.nanmedian(np.abs(sc.array("dsm_last") - truth)[trees]) < 0.2   # last returns hit ground
    before = sc.array("dtm_before")
    assert np.nanmedian((before - truth)[roof]) > 7.0     # planted roof error is in the "before" DTM
    assert sc.array("sem_ground")[roof].mean() > 0.9


def test_train_dataset_items(scene_dir):
    cfg = DataConfig(root=str(scene_dir.parent), tile=64, samples_per_epoch=40,
                     cond_channels=["dsm_max", "dsm_min", "dtm_before", "sem_ground", "density"],
                     norm_channels=["dsm_max", "dsm_min", "dtm_before"], prior_channel="dtm_before")
    ds = TileDataset(cfg, split=None, mode="train")
    ds.set_epoch(0)
    a = ds[3]
    assert a["cond"].shape == (5, 64, 64) and a["target"].shape == (1, 64, 64)
    assert torch.equal(ds[3]["target"], a["target"])        # deterministic per (epoch, index)
    ds.set_epoch(1)
    assert not torch.equal(ds[3]["target"], a["target"])    # different next epoch
    for i in range(40):
        it = ds[i]
        v = it["valid"][0] > 0
        assert torch.isfinite(it["cond"]).all() and torch.isfinite(it["target"]).all()
        # M_alpha must equal |s - g| < alpha recomputed from the returned tensors
        s_m = (it["cond"][0] + 1) * 0.5 * it["scale"] + it["lo"]
        g_m = (it["target"][0] + 1) * 0.5 * it["scale"] + it["lo"]
        close = (s_m - g_m).abs() < cfg.alpha
        mism = (close != (it["m_alpha"][0] > 0)) & v
        assert mism.float().mean() < 0.01      # float32 round-off at the threshold only


def test_eval_dataset_covers_scene(scene_dir):
    cfg = DataConfig(root=str(scene_dir.parent), tile=48, augment=False)
    ds = TileDataset(cfg, split=None, mode="eval")
    sc = ds.scenes[0]
    cover = np.zeros((sc.height, sc.width), int)
    for i in range(len(ds)):
        it = ds[i]
        r0, c0 = it["origin"].tolist()
        cover[r0:r0 + 48, c0:c0 + 48] += 1
    assert (cover > 0).all()


def test_block_split_is_spatial():
    centres = {f"s{i}_{j}": (i * 1000.0 + 500, j * 1000.0 + 500) for i in range(30) for j in range(30)}
    sp = block_split(centres, block_m=10_000.0, seed=0)
    blocks = {k: {(int(centres[n][0] // 10_000), int(centres[n][1] // 10_000)) for n in v}
              for k, v in sp.items()}
    assert not (blocks["train"] & blocks["val"]) and not (blocks["train"] & blocks["test"])
    assert sp["val"] and sp["test"] and sum(len(v) for v in sp.values()) == 900


def test_top_return_is():
    from groundiff.data.rasterise import top_return_is
    g = Grid.from_bounds(0, 0, 2, 1, 1.0)
    x = np.array([0.5, 0.5, 1.5]); y = np.array([0.5, 0.5, 0.5]); z = np.array([1.0, 5.0, 2.0])
    t = top_return_is(g, x, y, z, np.array([True, False, True]))
    assert t[0, 0] == 0.0 and t[0, 1] == 1.0          # highest return in cell 0 is non-ground


def test_top_class_m_alpha_and_fill(scene_dir):
    cfg = DataConfig(root=str(scene_dir.parent), tile=64, samples_per_epoch=10, m_alpha_mode="top_class",
                     fill_empty="nearest", cond_channels=["dsm_max", "dsm_min"], augment=False)
    ds = TileDataset(cfg, split=None, mode="train")
    it = ds[0]
    assert it["m_alpha"].sum() > 0 and torch.isfinite(it["cond"]).all()

"""Multi-year data (data.multiyear), Noise2Noise cross-year targets (dataset.PairScene) and the
Laplace label-noise head (losses, runtime)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from groundiff.data import multiyear as my
from groundiff.data.dataset import PairScene, Scene

N = 64


def _ground(seed=0):
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    return 20.0 + 0.05 * xx + 0.02 * yy


def _write(root: Path, loc: str, year: str, ground: np.ndarray, label: np.ndarray, building=None):
    sd = root / "scenes" / f"{loc}_{year}"
    sd.mkdir(parents=True)
    dsm = ground.copy()
    z_std = np.full((N, N), 0.02, np.float32)
    echoes = np.ones((N, N), np.float32)
    if building is not None:
        r0, r1, c0, c1 = building
        dsm[r0:r1, c0:c1] += 8.0
        z_std[r0:r1, c0:c1] = 0.5
    noise = np.random.default_rng(int(year)).normal(0, 0.03, (N, N)).astype(np.float32)
    arrs = {"dsm_min": dsm + noise, "dsm_last": dsm, "z_std": z_std, "echoes": echoes,
            "gt_dtm": label.astype(np.float32), "gt_valid": np.ones((N, N), np.float32)}
    for k, v in arrs.items():
        np.save(sd / f"{k}.npy", v.astype(np.float32))
    meta = {"scene": sd.name, "location": loc, "year": year,
            "grid": {"xmin": 0.0, "ymax": float(N), "width": N, "height": N, "gsd": 1.0}, "quality": {}}
    (sd / "meta.json").write_text(json.dumps(meta))
    return sd


def test_change_mask_dod():
    rng = np.random.default_rng(1)
    g = _ground()
    A = {"dsm_min": g + rng.normal(0, 0.06, g.shape).astype(np.float32)}
    b = g + 0.2 + rng.normal(0, 0.06, g.shape).astype(np.float32)    # 20 cm apart: below the LoD
    b[rng.random(g.shape) < 0.05] += 0.6                              # scattered outliers
    b[20:35, 20:35] += 3.0                                            # new ground (earthworks) in year B
    B = {"dsm_min": b}
    assert my.level_of_detection() == pytest.approx(0.416, abs=1e-3)
    um = my.unchanged_mask(A, B)
    assert not um[21:34, 21:34].any()                                 # the earthworks are masked
    far = np.ones_like(um)
    far[15:40, 15:40] = False
    assert um[far].mean() > 0.99                                      # scattered outliers are not
    B2 = {"dsm_min": np.where(np.arange(N)[None, :] < 32, b, np.nan)}
    assert not my.unchanged_mask(A, B2)[:, 40:].any()                 # surveyed in one year only


def test_pairs_consensus_removes_one_years_mistake(tmp_path):
    g = _ground()
    bad = g.copy()
    bad[40:50, 10:20] += 2.0                                # 2019's label left an object in
    _write(tmp_path, "L", "2017", g, g)
    _write(tmp_path, "L", "2019", g + 0.05, bad + 0.05)     # this survey sits 5 cm higher
    _write(tmp_path, "L", "2021", g, g, building=(5, 15, 45, 55))   # built on after 2019
    stats = my.pairs(tmp_path, log=lambda *a: None)
    assert stats == {"locations": 1, "pairs": 3, "consensus": 3}

    s19 = Scene(tmp_path / "scenes" / "L_2019")
    assert s19.meta["pairs"]["2017"]["lod"] == pytest.approx(0.416, abs=1e-3)
    cons = np.load(s19.path / "gt_consensus.npy")
    err = np.load(s19.path / "label_error.npy")
    # consensus is the true ground (within the surveys' few cm), the mistake outvoted by the other two years
    ok = np.isfinite(cons)
    assert np.nanmax(np.abs(cons - g)[ok]) < 0.06
    assert np.nanmin(err[42:48, 12:18]) > 1.9               # the mistake shows up as label error
    assert s19.meta["label_noise"]["frac_over_0.5m"] > 0
    # the 2021 building: those cells are masked out of pairs with 2021, so fewer than 3 years remain there
    assert not np.isfinite(cons[8:12, 48:52]).any()
    assert np.load(s19.path / "unchanged_2021.npy")[8:12, 48:52].max() == 0


def test_pair_scene_target_in_own_frame(tmp_path):
    g = _ground()
    _write(tmp_path, "L", "2017", g, g)
    _write(tmp_path, "L", "2019", g + 0.3, g + 0.3, building=(0, 10, 0, 10))
    my.pairs(tmp_path, min_consensus=3, log=lambda *a: None)
    a, b = Scene(tmp_path / "scenes" / "L_2017"), Scene(tmp_path / "scenes" / "L_2019")
    ps = PairScene(a, b, "2019")
    t = ps.window("gt_dtm", 0, 0, N, N)
    np.testing.assert_allclose(t[30:, 30:], g[30:, 30:] + 0.3, atol=1e-4)   # 2019's label as it is
    v = ps.window("gt_valid", 0, 0, N, N)
    assert v[2:8, 2:8].max() == 0 and v[40:, 40:].min() == 1           # changed cells carry no target
    np.testing.assert_array_equal(ps.window("dsm_min", 0, 0, N, N), a.window("dsm_min", 0, 0, N, N))
    with pytest.raises(KeyError):
        ps.array("gt_dtm")
    assert not (a.path / "gt_consensus.npy").exists()                  # only 2 years: no consensus


def test_clip_laz(tmp_path):
    import laspy
    from tests.synthetic import make_points, write_las
    x, y, z, cls, rn, nr = make_points(size=64.0, density=2.0)
    src = tmp_path / "in.laz"
    write_las(src, x, y, z, cls, rn, nr)
    box = (400010.0, 200010.0, 400030.0, 200030.0)
    n = my.clip_laz(src, tmp_path / "out.laz", [box], pad=5.0)
    las = laspy.read(tmp_path / "out.laz")
    assert len(las.points) == n > 0
    assert las.x.min() >= 400005.0 - 1e-6 and las.x.max() <= 400035.0 + 1e-6
    inside = (x >= 400005) & (x <= 400035) & (y >= 200005) & (y <= 200035)
    assert n == int(inside.sum())


def test_laplace_nll_optimum_and_gradients():
    from groundiff.losses import LossConfig, groundiff_loss
    e = 0.3
    g0 = torch.zeros(1, 1, 8, 8)
    pred = torch.full_like(g0, e)
    valid = torch.ones_like(g0)
    cfg = LossConfig(lam_l1=1.0, lam_l2=0.0, lam_grad=0.0, lam_conf=0.0, nll="laplace")
    vals = {}
    for b in (0.1, 0.3, 1.0):
        out = groundiff_loss(pred, torch.zeros_like(g0), g0, torch.zeros_like(g0), valid, cfg,
                             log_b=torch.full_like(g0, np.log(b)))
        vals[b] = float(out["loss"])
    assert vals[0.3] < vals[0.1] and vals[0.3] < vals[1.0]              # |e|/b + log b is minimised at b = |e|
    with pytest.raises(ValueError):
        groundiff_loss(pred, torch.zeros_like(g0), g0, torch.zeros_like(g0), valid, cfg)
    # the prediction's optimum is still the median (Noise2Noise with L1): gradient sign(e) / b
    p = pred.clone().requires_grad_(True)
    groundiff_loss(p, torch.zeros_like(g0), g0, torch.zeros_like(g0), valid, cfg,
                   log_b=torch.full_like(g0, np.log(0.5)))["loss"].backward()
    assert torch.allclose(p.grad, torch.full_like(g0, 1 / 0.5 / 64))


def _cfg(path):
    from groundiff.config import load_config
    return load_config(str(Path(__file__).parents[1] / path))


def test_aleatoric_model_and_init_from(tmp_path):
    from groundiff.models.build import build_model, init_from_checkpoint, save_checkpoint
    cfg2, cfg3 = _cfg("configs/n2n.json"), _cfg("configs/n2n_1step.json")
    for c in (cfg2, cfg3):
        c.model.base_channels, c.model.channel_mults = 16, [1, 2]
        c.model.attn_res, c.model.num_res_blocks = [], 1
    m2, m3 = build_model(cfg2), build_model(cfg3)
    torch.nn.init.normal_(m2.denoiser.out[2].weight)
    save_checkpoint(tmp_path / "a.pt", cfg2, m2.state_dict())
    notes = init_from_checkpoint(m3, cfg3, tmp_path / "a.pt")
    assert any("output head" in n for n in notes)
    w2, w3 = m2.denoiser.out[2].weight, m3.denoiser.out[2].weight
    assert w3.shape[0] == 3 and torch.equal(w3[:2], w2)
    x = torch.randn(2, 1 + len(cfg3.data.cond_channels), 32, 32)
    o = m3.training_forward(x[:, :1], x[:, 1:])
    assert o["log_b"] is not None and o["log_b"].shape == (2, 1, 32, 32)
    assert m2.training_forward(x[:, :1], x[:, 1:])["log_b"] is None


def test_runtime_noise_scale():
    from groundiff.batch import output_keys
    from groundiff.runtime import RuntimeSpec, predict_scene
    spec = RuntimeSpec.from_config(_cfg("configs/n2n_1step.json"))
    assert spec.aleatoric and "noise_scale" in output_keys(spec)
    t = spec.tile
    yy, xx = np.mgrid[0:t, 0:t].astype(np.float32)
    ground = 10.0 + 0.1 * xx                                 # 25.5 m range over the tile
    arrs = {"dsm_max": ground, "dsm_min": ground, "dsm_last": ground, "density": np.full_like(ground, 8.0),
            "z_std": np.full_like(ground, 0.02), "echoes": np.ones_like(ground)}
    lb = np.log(0.02)

    def net(x, gamma):
        out = np.zeros((x.shape[0], 3) + x.shape[2:], np.float32)
        out[:, 1] = 10.0                                     # keep the DSM
        out[:, 2] = lb
        return out
    res = predict_scene(arrs, spec, net, stride=t)
    assert np.nanmax(np.abs(res["dtm"] - ground)) < 0.05
    ns = res["noise_scale"]
    # b in metres = exp(log b) x half the tile's normalisation range
    from groundiff.runtime import tile_norm
    _, sc = tile_norm(arrs, spec)
    np.testing.assert_allclose(np.nanmean(ns), 0.02 * 0.5 * sc, rtol=1e-4)


def test_cross_year_sampling(tmp_path, monkeypatch):
    import groundiff.data.dataset as dsm
    g = _ground()
    for y, d in (("2017", 0.0), ("2019", 0.2), ("2021", 0.0)):
        sd = _write(tmp_path, "L", y, g + d, g + d)
        np.save(sd / "dsm_max.npy", (g + d).astype(np.float32))
        np.save(sd / "density.npy", np.full_like(g, 8.0))
    my.pairs(tmp_path, log=lambda *a: None)
    cfg = _cfg("configs/n2n.json").data
    cfg.tile = 32
    scenes = [Scene(p.parent) for p in sorted((tmp_path / "scenes").glob("*/meta.json"))]
    ds = dsm.TileDataset(cfg, None, mode="train", scenes=scenes)
    assert sum(len(v) for v in ds.partners.values()) == 6
    made = []
    real = dsm.PairScene
    monkeypatch.setattr(dsm, "PairScene", lambda *a: made.append(a[2]) or real(*a))
    rng = np.random.default_rng(0)
    for _ in range(60):
        arrs = ds._sample_train(rng)
        ok = np.isfinite(arrs["gt_dtm"]) & (arrs["gt_valid"] > 0.5)
        # own-year targets match the input; another year's differ by that survey's 0.2 m at most
        assert np.abs(arrs["gt_dtm"][ok] - arrs["dsm_max"][ok]).max() < 0.25
    assert 25 < len(made) < 55                              # about 2/3 of draws use another year's label
    ev = dsm.TileDataset(cfg, None, mode="eval", scenes=scenes)
    assert all(isinstance(s, dsm.ConsensusScene) for s in ev.scenes)


def test_prune_only_after_every_scene(tmp_path):
    sq = {"tile": "SU0000", "crops": [[0, 0, 2000, 2000], [2500, 0, 4500, 2000]]}
    for d in ("laz", "dtm"):
        f = tmp_path / d / "SU0000" / "2019"
        f.mkdir(parents=True)
        (f / "a.bin").write_text("x")
        (f / "DONE").write_text("1")
    first = tmp_path / "scenes" / f"{my.location_id('SU0000', sq['crops'][0])}_2019"
    first.mkdir(parents=True)
    (first / "meta.json").write_text("{}")
    assert not my._prune(tmp_path, sq, "2019") and (tmp_path / "laz/SU0000/2019/a.bin").exists()
    second = tmp_path / "scenes" / f"{my.location_id('SU0000', sq['crops'][1])}_2019"
    second.mkdir(parents=True)
    (second / "meta.json").write_text("{}")
    assert my._prune(tmp_path, sq, "2019")
    for d in ("laz", "dtm"):
        assert [p.name for p in (tmp_path / d / "SU0000" / "2019").iterdir()] == ["DONE"]


def test_clip_coverage_and_crop_choice(tmp_path, monkeypatch):
    from tests.synthetic import make_points, write_las
    x, y, z, cls, rn, nr = make_points(size=300.0, density=0.5, x0=400000.0, y0=200000.0)
    src = tmp_path / "in.laz"
    write_las(src, x, y, z, cls, rn, nr)
    boxes = [[400000, 200000, 400200, 200200], [400200, 200000, 400400, 200200], [401000, 201000, 401200, 201200]]
    cover = [np.zeros((2, 2), bool) for _ in boxes]
    my.clip_laz(src, tmp_path / "out.laz", boxes, pad=0.0, cover=cover)
    assert cover[0].all()                          # fully inside the 300 m of points
    assert cover[1][:, 0].all() and not cover[1][:, 1].any()   # half covered
    assert not cover[2].any()

    sq = {"tile": "SU0000", "bounds": [400000, 200000, 405000, 205000], "surveys": {"2017": {}, "2019": {}, "2021": {}},
          "crops": [[0, 0, 1, 1]] * 2}
    cov = {"2017": [1.0, 0.2, 0.0, 0.9], "2019": [1.0, 0.0, 0.0, 0.8], "2021": [0.5, 0.0, 0.0, 1.0]}
    for yr, c in cov.items():
        d = tmp_path / "laz" / "SU0000" / yr
        d.mkdir(parents=True)
        (d / "coverage.json").write_text(json.dumps({"cover": c}))
    dtm = {"2017": [1.0, 1.0, 1.0, 0.0], "2019": [1.0, 1.0, 1.0, 1.0], "2021": [1.0, 1.0, 1.0, 1.0]}
    monkeypatch.setattr(my, "dtm_coverage", lambda out, s, yr, b: dtm[yr])
    crops, scores = my.choose_crops(tmp_path, sq, 2)
    cand = my.candidate_boxes(sq)
    # crop 0: 1 + 1 + 0.5; crop 3: 0 (no DTM in 2017) + 0.8 + 1; crop 1: 0.2; crop 2: none
    assert crops == [cand[0], cand[3]] and scores == pytest.approx([2.5, 1.8])
    assert my.choose_crops(tmp_path, sq, 4)[0] == [cand[0], cand[3], cand[1]]


def test_rasterise_failure_is_isolated_and_remembered(tmp_path):
    sq = {"tile": "SU0000", "crops": [[400000, 200000, 402000, 202000]], "surveys": {"2019": {}}}
    loc, year, m, err = my._isolated((tmp_path, sq, sq["crops"][0], "2019", None))
    assert m is None and "no point files" in err
    name = f"{my.location_id('SU0000', sq['crops'][0])}_2019"
    assert my._scene_settled(tmp_path, name)                  # a deterministic failure is not retried
    d = tmp_path / "laz" / "SU0000" / "2019"
    d.mkdir(parents=True)
    (d / "a.laz").write_text("x")
    assert my._prune(tmp_path, sq, "2019") and not (d / "a.laz").exists()


def test_dtm_coverage_reads_rasters(tmp_path):
    from groundiff.io_raster import write_geotiff
    d = tmp_path / "dtm" / "SU0000" / "2019"
    d.mkdir(parents=True)
    a = 30.0 + 0.01 * np.add.outer(np.arange(2000), np.arange(2000)).astype(np.float32)   # sloping, not "water"
    a[:, 1000:] = np.nan                                     # the survey covers the west half
    write_geotiff(d / "dtm.tif", a, 400000.0, 202000.0, 1.0, None)
    sq = {"tile": "SU0000", "bounds": [400000, 200000, 405000, 205000]}
    cov = my.dtm_coverage(tmp_path, sq, "2019", [[400000, 200000, 402000, 202000], [402500, 200000, 404500, 202000]])
    assert cov[0] == pytest.approx(0.5, abs=0.02) and cov[1] == 0.0

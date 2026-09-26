import json

import numpy as np
import pytest

from groundiff.batch import Cancelled, plan, run_batch
from groundiff.data.laz import concat, read_points
from groundiff.data.rasterise import Grid, tin_dtm
from groundiff.runtime import RuntimeSpec
from tests.synthetic import make_points, write_las


@pytest.fixture(scope="module")
def tiles(tmp_path_factory):
    """One 160 m synthetic scene cut into 2x2 adjacent 80 m tiles, classified
    like lasground_new output (1/2 only, with a planted roof error), plus the
    same tiles with EA-style final classes (2, 5, 6, 7, 18)."""
    root = tmp_path_factory.mktemp("batch")
    x, y, z, cls, rn, nr = make_points(size=160.0, seed=5)
    before = np.where(cls == 2, 2, 1).astype(np.uint8)
    before[cls == 6] = 2                                  # planted lasground_new roof error
    x0, y0 = 400000.0, 200000.0
    after_files, before_files = [], []
    for i in range(2):
        for j in range(2):
            m = (x >= x0 + 80 * i) & (x < x0 + 80 * (i + 1)) & (y >= y0 + 80 * j) & (y < y0 + 80 * (j + 1))
            name = f"T{i}{j}.las"
            (root / "after").mkdir(exist_ok=True)
            (root / "before").mkdir(exist_ok=True)
            write_las(root / "after" / name, x[m], y[m], z[m], cls[m], rn[m], nr[m])
            write_las(root / "before" / name, x[m], y[m], z[m], before[m], rn[m], nr[m])
            after_files.append(root / "after" / name)
            before_files.append(root / "before" / name)
    return root, after_files, before_files


def spec_before_after(tile=32):
    from groundiff.schedule import build_schedule
    s = build_schedule("cosine", 10)
    return RuntimeSpec(kind="groundiff", cond_channels=["dsm_max", "dsm_min", "dtm_before", "sem_ground"],
                       gate_channel="dtm_before", norm_channels=["dsm_max", "dsm_min", "dtm_before"],
                       prior_channel="dtm_before", norm_mode="minmax", norm_std=None, min_range=2.0, tile=tile,
                       alpha=0.2, T=10, alphas_bar=s.alphas_bar.tolist(), coef_x0=s.coef_x0.tolist(),
                       coef_xt=s.coef_xt.tolist(), posterior_var=s.posterior_var.tolist(), fill_empty="nearest",
                       gsd=1.0)


def keep_gate_net(x, gamma):
    """Stand-in network: r_hat = 0 and a confident 'keep' logit, so the
    predicted DTM equals the gate surface (the lasground_new DTM)."""
    b, _, h, w = x.shape
    return np.concatenate([np.zeros((b, 1, h, w), np.float32), np.full((b, 1, h, w), 30.0, np.float32)], 1)


def noisy_net(x, gamma):
    """Output depends on the noisy state g_t (channel 0), so any difference in
    sampling noise between jobs would show up as a seam."""
    b, _, h, w = x.shape
    return np.concatenate([0.3 * x[:, :1] + 0.1 * x[:, 1:2], np.zeros((b, 1, h, w), np.float32)], 1).astype(np.float32)


def read(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read(1, masked=True).filled(np.nan), src.transform


def test_plan_neighbours(tiles):
    _, _, before = tiles
    jobs, union, problems = plan(before, gsd=1.0, buffer_m=16.0)
    assert not problems
    assert len(jobs) == 4 and all(len(j.files) == 4 for j in jobs)        # 2x2: every tile touches all others
    assert union[2] - union[0] == pytest.approx(160.0, abs=1.0)
    jobs2, _, _ = plan(before, gsd=1.0, buffer_m=16.0, max_block_m=40.0)
    assert len(jobs2) == 16                                                 # each 80 m tile split in 2x2 blocks


def test_plan_rejects_duplicate_names(tiles, tmp_path):
    _, after, before = tiles
    with pytest.raises(ValueError, match="more than once"):
        plan([before[0], after[0]], gsd=1.0, buffer_m=16.0)


def test_plan_repairs_stale_header(tiles, tmp_path):
    import laspy
    _, _, before = tiles
    las = laspy.read(str(before[0]))
    bad = tmp_path / "T00.las"
    las.write(str(bad))
    with open(bad, "r+b") as f:             # LAS 1.4 header: max X at byte 179, min X at 187 (float64)
        f.seek(179)
        f.write(np.float64(400000.5).tobytes())
    msgs = []
    jobs, _, _ = plan([bad], gsd=1.0, buffer_m=16.0, log=msgs.append)
    assert any("look wrong" in m for m in msgs)
    assert jobs[0].core[2] - jobs[0].core[0] == pytest.approx(80.0, abs=1.0)


def test_batch_mosaic_is_seamless(tiles, tmp_path):
    root, _, before = tiles
    spec = spec_before_after()
    s = run_batch(before, tmp_path / "out", keep_gate_net, spec, buffer_m=40.0, workers=2,
                  predict_kwargs={"batch_size": 4})
    mosaic, tr = read(tmp_path / "out" / "dtm.tif")
    # reference: TIN of ALL lasground_new ground points on the same global grid
    pts = concat([read_points(p) for p in before])
    g = pts.cls == 2
    G = Grid(tr.c, tr.f, 1.0, mosaic.shape[1], mosaic.shape[0])
    ref, ok = tin_dtm(G, pts.x[g], pts.y[g], pts.z[g])
    inner = ok.copy()
    inner[:3], inner[-3:], inner[:, :3], inner[:, -3:] = False, False, False, False
    diff = np.abs(mosaic - ref)[inner & np.isfinite(mosaic)]
    assert diff.size > 0.9 * inner.sum()
    assert diff.max() < 1e-3                                                # identical across tile seams
    summary = json.loads((tmp_path / "out" / "batch_summary.json").read_text())
    assert len(summary["tiles"]) == 4 and not summary["failed"]
    for f in ("p_edit.tif", "dz_before.tif", "p_edit_overlay.tif", "p_edit_overlay_rgb.tif", "p_edit.qml",
              "dz_before.qml", "dtm.vrt", "p_edit_overlay.tfw", "dtm.prj", "priority.csv", "priority.geojson"):
        assert (tmp_path / "out" / f).exists(), f
    import rasterio
    with rasterio.open(tmp_path / "out" / "p_edit_overlay.tif") as src:
        assert src.count == 4 and src.read(4).max() == 0                     # p_edit ~ 0: fully transparent
    gj = json.loads((tmp_path / "out" / "priority.geojson").read_text())
    assert gj["features"] and gj["features"][0]["properties"]["rank"] == 1
    assert gj["crs"]["properties"]["name"].endswith("27700")


def test_stochastic_outputs_do_not_depend_on_job_split(tiles, tmp_path):
    """Global tile lattice + per-tile seeds: splitting the area into more
    jobs must not change any value."""
    _, _, before = tiles
    spec = spec_before_after()
    kw = {"batch_size": 3, "init": "prior_noise", "seed": 7}
    run_batch(before, tmp_path / "a", noisy_net, spec, buffer_m=72.0, workers=1, predict_kwargs=dict(kw),
              overlays=False)
    run_batch(before, tmp_path / "b", noisy_net, spec, buffer_m=72.0, workers=1, predict_kwargs=dict(kw),
              overlays=False, max_block_m=40.0)
    a, _ = read(tmp_path / "a" / "dtm.tif")
    b, _ = read(tmp_path / "b" / "dtm.tif")
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() > 0.9 * a.size
    assert np.abs(a - b)[both].max() < 1e-4
    c, _ = read(tmp_path / "a" / "dz_before.tif")
    assert np.nanstd(c) > 0.01                                              # the net really is noisy


def test_batch_rejects_tiles_not_from_lasground(tiles, tmp_path):
    _, after, _ = tiles
    with pytest.raises(ValueError, match="lasground_new"):
        run_batch(after, tmp_path / "x", keep_gate_net, spec_before_after(), buffer_m=16.0)
    summary = json.loads((tmp_path / "x" / "batch_summary.json").read_text()) if \
        (tmp_path / "x" / "batch_summary.json").exists() else None
    assert summary is None or summary["failed"]


def test_batch_cancel(tiles, tmp_path):
    _, _, before = tiles
    calls = {"n": 0}

    def cancelled():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(Cancelled):
        run_batch(before, tmp_path / "c", keep_gate_net, spec_before_after(), buffer_m=16.0,
                  cancelled=cancelled)
    s = json.loads((tmp_path / "c" / "batch_summary.json").read_text())
    assert s["cancelled"] and not (tmp_path / "c" / "dtm.tif").exists()


def spec_dsm_only(tile=32):
    from groundiff.schedule import build_schedule
    s = build_schedule("cosine", 10)
    return RuntimeSpec(kind="groundiff", cond_channels=["dsm_max", "dsm_min", "density"], gate_channel="dsm_max",
                       norm_channels=["dsm_max", "dsm_min"], prior_channel=None, norm_mode="minmax", norm_std=None,
                       min_range=2.0, tile=tile, alpha=0.2, T=10, alphas_bar=s.alphas_bar.tolist(),
                       coef_x0=s.coef_x0.tolist(), coef_xt=s.coef_xt.tolist(), posterior_var=s.posterior_var.tolist(),
                       fill_empty="nearest", gsd=1.0)


def test_dsm_only_model_gives_edit_map_on_lasground_tiles(tiles, tmp_path):
    """A model that never saw lasground_new: on lasground_new tiles the batch
    compares every sample with their ground -> dz_before and sampled p_edit.
    Stand-in net keeps the DSM, so trees (canopy vs lasground ground) need an
    edit and open ground does not."""
    _, after, before = tiles
    spec = spec_dsm_only()
    run_batch(before, tmp_path / "o", keep_gate_net, spec, buffer_m=40.0, workers=1,
              predict_kwargs={"batch_size": 4, "n_samples": 2})
    pe, tr = read(tmp_path / "o" / "p_edit.tif")
    dz, _ = read(tmp_path / "o" / "dz_before.tif")
    xs = tr.c + (np.arange(pe.shape[1]) + 0.5)
    ys = tr.f - (np.arange(pe.shape[0]) + 0.5)
    X, Y = np.meshgrid(xs - 400000.0, ys - 200000.0)
    trees = (X > 85) & (X < 105) & (Y > 75) & (Y < 105)
    open_ground = (X > 120) & (X < 150) & (Y > 120) & (Y < 150)
    assert np.nanmean(pe[trees]) > 0.8 and np.nanmean(dz[trees]) > 5.0
    assert np.nanmean(pe[open_ground]) < 0.1
    assert (tmp_path / "o" / "p_edit_overlay.tif").exists() and (tmp_path / "o" / "std.tif").exists()
    # published-style classes: no lasground reference -> DTM only, no error
    s = run_batch(after, tmp_path / "p", keep_gate_net, spec, buffer_m=40.0, workers=1,
                  predict_kwargs={"batch_size": 4})
    assert "dtm" in s["outputs"] and "p_edit" not in s["outputs"] and "dz_before" not in s["outputs"]

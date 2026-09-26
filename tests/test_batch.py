import json

import numpy as np
import pytest

from groundiff.batch import plan, run_batch
from groundiff.data.laz import concat, read_points
from groundiff.data.rasterise import Grid, tin_dtm
from groundiff.runtime import RuntimeSpec
from tests.synthetic import make_points, write_las


@pytest.fixture(scope="module")
def tiles(tmp_path_factory):
    """One 160 m synthetic scene cut into 2x2 adjacent 80 m tiles (after + before)."""
    root = tmp_path_factory.mktemp("batch")
    x, y, z, cls, rn, nr = make_points(size=160.0, seed=5)
    before = np.where(cls == 2, 2, 1).astype(np.uint8)
    before[np.isin(cls, (7, 18))] = cls[np.isin(cls, (7, 18))]
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
                       coef_xt=s.coef_xt.tolist(), posterior_var=s.posterior_var.tolist(), fill_empty="nearest")


def keep_gate_net(x, gamma):
    """Stand-in network: r_hat = 0 and a confident 'keep' logit, so the
    predicted DTM equals the gate surface (the lasground_new DTM)."""
    b, _, h, w = x.shape
    return np.concatenate([np.zeros((b, 1, h, w), np.float32), np.full((b, 1, h, w), 30.0, np.float32)], 1)


def test_plan_neighbours(tiles):
    _, after, before = tiles
    jobs, union = plan(after, before, gsd=1.0, buffer_m=16.0)
    assert len(jobs) == 4 and all(len(j.after_files) == 4 for j in jobs)   # 2x2: every tile touches all others
    assert union[2] - union[0] == pytest.approx(160.0, abs=1.0)
    jobs2, _ = plan(after, before, gsd=1.0, buffer_m=16.0, max_block_m=40.0)
    assert len(jobs2) == 16                                                 # each 80 m tile split in 2x2 blocks


def test_batch_mosaic_is_seamless(tiles, tmp_path):
    import rasterio
    root, after, before = tiles
    spec = spec_before_after()
    s = run_batch(after, tmp_path / "out", keep_gate_net, spec, before_files=before, gsd=1.0, buffer_m=16.0,
                  workers=2, predict_kwargs={"batch_size": 4})
    with rasterio.open(tmp_path / "out" / "dtm.tif") as src:
        mosaic = src.read(1, masked=True).filled(np.nan)
        tr = src.transform
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
    assert len(summary["tiles"]) == 4
    for f in ("p_edit.tif", "dz_before.tif", "p_edit_overlay.tif", "p_edit_overlay_rgb.tif", "p_edit.qml", "dtm.vrt",
              "p_edit_overlay.tfw"):
        assert (tmp_path / "out" / f).exists(), f
    with rasterio.open(tmp_path / "out" / "p_edit_overlay.tif") as src:
        assert src.count == 4 and src.read(4).max() == 0                     # p_edit ~ 0: fully transparent


def test_batch_requires_before_for_before_after_models(tiles, tmp_path):
    _, after, _ = tiles
    with pytest.raises(ValueError):
        run_batch(after, tmp_path / "x", keep_gate_net, spec_before_after())

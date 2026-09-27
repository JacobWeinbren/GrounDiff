import numpy as np
import pytest

from groundiff.data.laz import read_points_bbox
from groundiff.data.rasterise import Grid, input_rasters, tin_dtm_local, _tin_cells
from groundiff.data.stream import Accumulator, stream_file
from tests.synthetic import make_points, write_las


def test_stream_matches_in_memory_rasters(tmp_path):
    x, y, z, cls, rn, nr = make_points(size=96.0, seed=3)
    cls = np.where(cls == 2, 2, 1).astype(np.uint8)              # lasground_new's 1 / 2
    rng = np.random.default_rng(0)
    nr = rng.integers(1, 4, x.size).astype(np.uint8)
    rn = np.minimum(rng.integers(1, 4, x.size), nr).astype(np.uint8)
    f = tmp_path / "t.las"
    write_las(f, x, y, z, cls, rn, nr)
    b = (x.min() + 10, y.min() + 5, x.max() - 7, y.max() - 3)
    g = Grid.from_bounds(*b, 1.0)
    bbox = g.bounds
    ref = input_rasters(g, read_points_bbox(f, bbox), True)
    acc = Accumulator(g, keep_ground=True)
    seen = []
    stream_file(f, bbox, acc, chunk_size=997, progress=seen.append)   # many chunks, cells split across them
    got = acc.rasters(True)
    assert seen and seen[-1] == pytest.approx(1.0)
    assert set(got) == set(ref)
    for k in ref:
        a, r = got[k], ref[k]
        assert np.array_equal(np.isnan(a), np.isnan(r)), k
        tol = 1e-4 if k in ("z_std", "dtm_before") else 0.0      # float32 ground coords / merged variance
        assert np.nanmax(np.abs(a - r)) <= tol, (k, np.nanmax(np.abs(a - r)))


def test_blocked_tin_equals_one_tin():
    """Blocks + circumcircle check give exactly the single triangulation, voids included."""
    rng = np.random.default_rng(1)
    S = 300.0
    n = int(S * S * 2)
    x, y = rng.random(n) * S, rng.random(n) * S
    keep = ~(((x - 100) ** 2 + (y - 150) ** 2) < 45 ** 2) & ~((x > 200) & (x < 235) & (y > 20) & (y < 280))
    x, y = x[keep], y[keep]
    z = np.sin(x / 40) * 3 + np.cos(y / 25) * 2 + rng.random(x.size) * 0.1
    g = Grid(0.0, S, 1.0, int(S), int(S))
    lx, ly = x, y - S
    ref = np.full((g.height, g.width), np.nan, np.float32)
    _tin_cells(ref, g, lx, ly, z, (0, g.height, 0, g.width), None, (lx.min(), ly.min(), lx.max(), ly.max()))
    got, valid = tin_dtm_local(g, lx, ly, z, block_points=8000, workers=3)
    assert np.array_equal(np.isfinite(ref), valid)
    assert np.nanmax(np.abs(ref - got)) == 0.0


@pytest.mark.parametrize("use_triangle", [True, False])
def test_tin_matches_scipy_linear_interpolator(monkeypatch, use_triangle):
    """Same values as the LinearNDInterpolator TIN it replaced (random points: no cocircular ties)."""
    import builtins
    from scipy.interpolate import LinearNDInterpolator
    from groundiff.data.rasterise import tin_dtm
    if not use_triangle:
        real = builtins.__import__
        monkeypatch.setattr(builtins, "__import__",
                            lambda name, *a, **k: (_ for _ in ()).throw(ImportError()) if name == "triangle"
                            else real(name, *a, **k))
    elif pytest.importorskip("triangle") is None:
        return
    rng = np.random.default_rng(4)
    x, y = 400000 + rng.random(20000) * 150, 200000 + rng.random(20000) * 120
    z = 30 + np.sin(x / 9) + rng.random(x.size) * 0.2
    g = Grid.from_bounds(400000 - 5, 200000 - 5, 400155, 200125, 1.0)
    got, valid = tin_dtm(g, x, y, z, block_points=3000, workers=2)
    xs, ys = g.cell_centres()
    XX, YY = np.meshgrid(xs - g.xmin, ys - g.ymax)
    ref = LinearNDInterpolator(np.column_stack([x - g.xmin, y - g.ymax]), z, fill_value=np.nan)(XX, YY)
    assert np.array_equal(np.isfinite(ref), valid)
    assert np.nanmax(np.abs(ref.astype(np.float32) - got)) <= 4e-6        # float32 output, as before


def test_tin_with_big_void_is_exact_and_bounded():
    """A ring of points around a large empty square: every triangle inside spans the void.
    The scanline lookup keeps the work to the cells covered (a bounding-box scan of these
    slivers was what used tens of GB on real tiles), and blocks still equal one TIN."""
    import time
    from groundiff.data.rasterise import _locate
    rng = np.random.default_rng(5)
    S = 800.0
    n = int(S * S * 1.5)
    x, y = rng.random(n) * S, rng.random(n) * S
    keep = ~((x > 60) & (x < 740) & (y > 60) & (y < 740))
    x, y = np.round(x[keep], 2), np.round(y[keep], 2)                 # cm, as EA tiles
    z = np.round(20 + x * 0.01 + rng.random(x.size) * 0.05, 2)
    x, y, z = np.r_[x, x[:50]], np.r_[y, y[:50]], np.r_[z, z[:50] + 0.03]   # duplicate x, y with other z
    g = Grid(0.0, S, 1.0, int(S), int(S))
    lx, ly = x, y - S
    ext = (lx.min(), ly.min(), lx.max(), ly.max())
    t0 = time.time()
    got, valid = tin_dtm_local(g, lx, ly, z, block_points=20_000, workers=2)
    assert time.time() - t0 < 120
    ref = np.full((g.height, g.width), np.nan, np.float32)
    _tin_cells(ref, g, lx, ly, z, (0, g.height, 0, g.width), None, ext)
    assert np.array_equal(np.isfinite(ref), valid)
    assert np.nanmax(np.abs(ref - got)) == 0.0
    # thin triangles: work stays near the number of cells
    pts = np.array([[0.0, 0.0], [700.0, -1.0], [699.0, -700.0], [1.0, -699.0]])
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    rr, cc = np.meshgrid(np.arange(700), np.arange(700), indexing="ij")
    ci, ti, w = _locate(pts, tris, rr.ravel(), cc.ravel(), 1.0, max_items=1000)
    assert ci.size > 480_000 and np.allclose(w.sum(1), 1.0)

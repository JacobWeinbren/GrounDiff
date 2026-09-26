"""The reader and inspector must cope with the ways producers' LAS files differ
(EA deliveries vs tiles saved by editing software such as LP360)."""
import numpy as np
import pytest

from groundiff.data.lasinspect import compare, inspect
from groundiff.data.laz import read_points


def _write(path, version, fmt, n=200, extra=False, wkt_bit=None, setup=None):
    import laspy
    rng = np.random.default_rng(0)
    hdr = laspy.LasHeader(point_format=fmt, version=version)
    if extra:
        hdr.add_extra_dim(laspy.ExtraBytesParams(name="editor_flag", type=np.uint8))
    hdr.offsets = [400000.0, 200000.0, 0.0]
    hdr.scales = [0.01, 0.01, 0.01]
    if wkt_bit is not None:
        hdr.global_encoding.wkt = wkt_bit
    las = laspy.LasData(hdr)
    las.x = 400000 + rng.uniform(0, 50, n)
    las.y = 200000 + rng.uniform(0, 50, n)
    las.z = rng.uniform(10, 20, n)
    las.classification = np.full(n, 2, np.uint8)
    las.return_number = np.ones(n, np.uint8)
    las.number_of_returns = np.ones(n, np.uint8)
    if setup:
        setup(las)
    las.write(str(path))
    return path


def test_legacy_las12_laz(tmp_path):
    p = _write(tmp_path / "legacy.laz", "1.2", 1)
    pts = read_points(p)
    assert len(pts) == 200
    rep = inspect(p)
    assert rep["version"] == "1.2" and rep["point_format"] == 1


def test_las14_without_wkt_bit_is_read_and_flagged(tmp_path):
    p = _write(tmp_path / "nowkt.las", "1.4", 6, wkt_bit=False)
    assert len(read_points(p)) == 200
    assert any("WKT bit" in w for w in inspect(p)["warnings"])


def test_extra_bytes(tmp_path):
    p = _write(tmp_path / "extra.las", "1.4", 7, extra=True, wkt_bit=True)
    assert len(read_points(p)) == 200
    assert inspect(p)["extra_dimensions"] == ["editor_flag"]


def test_inconsistent_returns(tmp_path):
    def setup(las):
        rn = np.ones(200, np.uint8); nr = np.ones(200, np.uint8)
        rn[:10] = 3; nr[:10] = 2          # return_number > number_of_returns
        nr[10:20] = 0                     # number_of_returns = 0
        rn[20:30] = 0                     # return_number = 0
        las.return_number = rn
        las.number_of_returns = nr
    p = _write(tmp_path / "returns.las", "1.4", 6, wkt_bit=True, setup=setup)
    pts = read_points(p)
    assert pts.number_of_returns.min() >= 1 and pts.return_number.min() >= 1
    r = inspect(p)["returns"]
    assert r["return_gt_number"] == 10 and r["number_of_returns_0"] == 10 and r["return_number_0"] == 10


def test_flags_and_overlap(tmp_path):
    def setup(las):
        cls = np.full(200, 2, np.uint8)
        cls[:5] = 7                                   # noise
        las.classification = cls
        ov = np.zeros(200, bool); ov[5:15] = True
        syn = np.zeros(200, bool); syn[15:20] = True
        wh = np.zeros(200, bool); wh[20:22] = True
        las.overlap, las.synthetic, las.withheld = ov, syn, wh
    p = _write(tmp_path / "flags.las", "1.4", 6, wkt_bit=True, setup=setup)
    assert len(read_points(p)) == 200 - 5 - 2                       # noise + withheld dropped by default
    assert len(read_points(p, drop_overlap=True, drop_synthetic=True)) == 200 - 5 - 2 - 10 - 5
    rep = inspect(p)
    assert rep["flags"]["overlap"] == 10 and rep["flags"]["synthetic"] == 5
    assert any("overlap" in w for w in rep["warnings"]) and any("synthetic" in w for w in rep["warnings"])


def test_legacy_overlap_class(tmp_path):
    def setup(las):
        cls = np.full(200, 2, np.uint8); cls[:8] = 12
        las.classification = cls
    p = _write(tmp_path / "c12.las", "1.2", 3, setup=setup)
    assert len(read_points(p, drop_overlap=True)) == 192


def test_points_outside_header_bounds_are_flagged(tmp_path):
    import laspy
    p = _write(tmp_path / "bounds.las", "1.4", 6, wkt_bit=True)
    las = laspy.read(str(p))
    las.header.maxs = las.header.maxs - 10.0          # stale header, as after editing
    las.write(str(tmp_path / "bounds2.las"), do_compress=False)
    # laspy recomputes bounds on write, so patch the header bytes directly
    raw = bytearray((tmp_path / "bounds2.las").read_bytes())
    import struct
    off = 179                                          # max X in the LAS header
    struct.pack_into("<d", raw, off, struct.unpack_from("<d", raw, off)[0] - 10.0)
    (tmp_path / "bounds3.las").write_bytes(bytes(raw))
    assert any("outside the header bounds" in w for w in inspect(tmp_path / "bounds3.las")["warnings"])
    assert len(read_points(tmp_path / "bounds3.las")) == 200


def test_compare_reports_differences(tmp_path):
    a = inspect(_write(tmp_path / "a.laz", "1.2", 1))
    b = inspect(_write(tmp_path / "b.las", "1.4", 7, extra=True, wkt_bit=False))
    d = compare(a, b)
    assert "version" in d and "point_format" in d and "extra_dimensions" in d

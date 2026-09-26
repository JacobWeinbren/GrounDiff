"""Inspect LAS/LAZ/COPC files and flag things that trip up other software.

    python -m groundiff.data.lasinspect tile.laz
    python -m groundiff.data.lasinspect ea_tile.laz --compare lp360_tile.las

Files from different producers (e.g. EA deliveries vs tiles saved by LP360)
differ in ways that some readers reject. The known ones are checked here:
  * LAS 1.4 point formats 6-10 without the global-encoding WKT bit (the spec
    requires it; PDAL, and therefore QGIS point-cloud layers, refuse such files)
  * CRS stored as GeoTIFF keys, as WKT, or missing
  * point records longer than the format without an Extra Bytes VLR
  * legacy point count / header count mismatches
  * points outside the header bounds
  * return numbers of 0, number_of_returns of 0, return_number > number_of_returns
  * overlap points (LAS 1.4 overlap flag or legacy class 12), synthetic,
    key-point and withheld flags
  * non-standard classes
This reader (laspy) tolerates all of these; the report tells you which
options to set when rasterising.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

STANDARD_CLASSES = {0: "created, never classified", 1: "unclassified", 2: "ground", 3: "low vegetation",
                    4: "medium vegetation", 5: "high vegetation", 6: "building", 7: "low point (noise)",
                    8: "model key-point / reserved", 9: "water", 10: "rail", 11: "road surface",
                    12: "overlap (legacy) / reserved", 13: "wire guard", 14: "wire conductor",
                    15: "transmission tower", 16: "wire connector", 17: "bridge deck", 18: "high noise",
                    19: "overhead structure", 20: "ignored ground", 21: "snow", 22: "temporal exclusion"}
STANDARD_RECORD_LEN = {0: 20, 1: 28, 2: 26, 3: 34, 4: 57, 5: 63, 6: 30, 7: 36, 8: 38, 9: 59, 10: 67}


def _flag(las, name):
    try:
        return np.asarray(getattr(las, name), dtype=bool)
    except Exception:
        return None


def inspect(path: str | Path, max_points: int | None = None) -> dict:
    import laspy

    path = Path(path)
    rep: dict = {"file": str(path), "size_mb": round(path.stat().st_size / 1e6, 2), "warnings": []}
    warn = rep["warnings"].append
    with laspy.open(str(path)) as f:
        h = f.header
        rep.update({
            "version": f"{h.version.major}.{h.version.minor}",
            "point_format": h.point_format.id,
            "record_length": h.point_format.size,
            "extra_dimensions": list(h.point_format.extra_dimension_names),
            "point_count": int(h.point_count),
            "scales": [float(v) for v in h.scales], "offsets": [float(v) for v in h.offsets],
            "header_mins": [float(v) for v in h.mins], "header_maxs": [float(v) for v in h.maxs],
            "system_identifier": str(getattr(h, "system_identifier", "")).strip("\x00 "),
            "generating_software": str(getattr(h, "generating_software", "")).strip("\x00 "),
            "vlrs": [f"{v.user_id.strip()}:{v.record_id} {getattr(v, 'description', '')}".strip() for v in h.vlrs],
            "evlrs": [f"{v.user_id.strip()}:{v.record_id}" for v in (getattr(h, "evlrs", None) or [])],
        })
        ge = getattr(h, "global_encoding", None)
        wkt_bit = bool(getattr(ge, "wkt", False)) if ge is not None else False
        rep["global_encoding_wkt"] = wkt_bit
        is_copc = any(v.user_id.strip() == "copc" for v in h.vlrs)
        rep["copc"] = is_copc
        las = f.read_points(max_points) if max_points else f.read_points(h.point_count)
    pf = rep["point_format"]
    if pf >= 6 and not wkt_bit:
        warn("point format >= 6 but the global-encoding WKT bit is not set: PDAL/QGIS may refuse this "
             "file (read it by path with laspy, or rewrite it with `las2las -set_global_encoding_wkt` "
             "or PDAL writers.las a_srs)")
    crs_vlrs = [v for v in rep["vlrs"] if v.startswith(("LASF_Projection:2112", "LASF_Projection:34735"))]
    rep["crs_storage"] = ("WKT" if any(":2112" in v for v in crs_vlrs) else
                          "GeoTIFF keys" if any(":34735" in v for v in crs_vlrs) else "none")
    try:
        crs = h.parse_crs()
        rep["crs"] = crs.name if crs is not None else None
    except Exception as e:     # pyproj missing or malformed VLR
        rep["crs"] = None
        rep["crs_error"] = repr(e)
    if rep["crs_storage"] == "none":
        warn("no CRS VLR: outputs will have no CRS unless you set one")
    if pf >= 6 and rep["crs_storage"] == "GeoTIFF keys":
        warn("point format >= 6 with a GeoTIFF-key CRS (LAS 1.4 requires WKT for these formats)")
    std_len = STANDARD_RECORD_LEN.get(pf)
    if std_len and rep["record_length"] > std_len and not rep["extra_dimensions"]:
        warn(f"records are {rep['record_length']} bytes (format {pf} is {std_len}) but no Extra Bytes VLR "
             "describes the extra bytes")

    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    n = x.size
    rep["n_read"] = int(n)
    if n and max_points is None and n != rep["point_count"]:
        warn(f"header point count {rep['point_count']} but {n} points read")
    if n:
        dmin = [float(x.min()), float(y.min()), float(z.min())]
        dmax = [float(x.max()), float(y.max()), float(z.max())]
        rep["data_mins"], rep["data_maxs"] = dmin, dmax
        tol = [s * 1.01 for s in rep["scales"]]
        if any(d < hm - t for d, hm, t in zip(dmin, rep["header_mins"], tol)) or \
           any(d > hm + t for d, hm, t in zip(dmax, rep["header_maxs"], tol)):
            warn("points lie outside the header bounds (header not updated after editing?)")
    cls = np.asarray(las.classification, dtype=np.int64)
    vals, counts = np.unique(cls, return_counts=True)
    rep["classes"] = {int(v): {"count": int(c), "name": STANDARD_CLASSES.get(int(v), "non-standard")}
                      for v, c in zip(vals, counts)}
    odd = [int(v) for v in vals if int(v) not in STANDARD_CLASSES]
    if odd:
        warn(f"non-standard classes present: {odd}")
    rn = np.asarray(las.return_number, dtype=np.int64)
    nr = np.asarray(las.number_of_returns, dtype=np.int64)
    rep["returns"] = {"return_number_0": int((rn == 0).sum()), "number_of_returns_0": int((nr == 0).sum()),
                      "return_gt_number": int(((rn > nr) & (nr > 0)).sum()), "max_number_of_returns": int(nr.max()) if n else 0}
    if rep["returns"]["return_number_0"] or rep["returns"]["number_of_returns_0"]:
        warn("points with return_number 0 or number_of_returns 0 (treated as single returns)")
    if rep["returns"]["return_gt_number"]:
        warn(f"{rep['returns']['return_gt_number']} points have return_number > number_of_returns "
             "(treated as last returns)")
    flags = {}
    for name in ("withheld", "overlap", "synthetic", "key_point"):
        fl = _flag(las, name)
        if fl is not None:
            flags[name] = int(fl.sum())
    rep["flags"] = flags
    if flags.get("overlap") or 12 in rep["classes"]:
        warn("overlap points present (flag or class 12); consider --drop-overlap")
    if flags.get("synthetic"):
        warn(f"{flags['synthetic']} synthetic points (e.g. added by editing software); consider --drop-synthetic")
    if flags.get("withheld"):
        warn(f"{flags['withheld']} withheld points (dropped by default)")
    return rep


COMPARE_KEYS = ("version", "point_format", "record_length", "extra_dimensions", "scales", "offsets",
                "global_encoding_wkt", "crs_storage", "crs", "copc", "system_identifier",
                "generating_software", "vlrs", "evlrs", "flags", "returns")


def compare(a: dict, b: dict) -> dict:
    diff = {k: (a.get(k), b.get(k)) for k in COMPARE_KEYS if a.get(k) != b.get(k)}
    ca, cb = set(a.get("classes", {})), set(b.get("classes", {}))
    if ca != cb:
        diff["classes_only_in_first"] = sorted(ca - cb)
        diff["classes_only_in_second"] = sorted(cb - ca)
    return diff


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--compare", help="second file to diff against")
    ap.add_argument("--max-points", type=int, help="only read this many points (quick look)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = inspect(a.file, a.max_points)
    out = {"file": rep}
    if a.compare:
        rep2 = inspect(a.compare, a.max_points)
        out = {"first": rep, "second": rep2, "differences": compare(rep, rep2)}
    if a.json:
        print(json.dumps(out, indent=1, default=str))
    else:
        for key, r in (out.items() if a.compare else [("file", rep)]):
            if key == "differences":
                print("\n== differences (first, second)")
                for k, v in r.items():
                    print(f"  {k}: {v[0] if isinstance(v, tuple) else v}  |  {v[1] if isinstance(v, tuple) else ''}")
                continue
            print(f"\n== {r['file']}")
            for k, v in r.items():
                if k not in ("file", "warnings"):
                    print(f"  {k}: {v}")
            for w in r["warnings"]:
                print(f"  WARNING: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

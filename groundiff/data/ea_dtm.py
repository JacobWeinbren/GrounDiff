"""Download the EA's published DTM rasters (the training target) for a set of
point-cloud tiles.

Source: the Defra Data Services Platform "survey" service
(https://environment.data.gov.uk/survey), no login, the same two calls its
web page makes:

  search    POST .../backend/catalog/api/tiles/collections/survey/search
            (GeoJSON polygon in lon/lat, Content-Type application/geo+json)
  download  GET  .../tiles/collections/survey/{product}/{year}/{res}/{tile}?subscription-key=dspui

"dspui" is the public key built into that web page (not a credential); it is
undocumented and may change. Tiles are 5 km OS squares (id "TL4075" = south-
west corner in km, label "TL47nw"); each zip holds GeoTIFFs in EPSG:27700
(ODN heights, nodata -3.4e38; 5000 x 5000 at 1 m) and a metadata GeoPackage
saying which survey fed each area.

Which product: the DEFRA LIDAR_YYYY point clouds (s3://open-lidar-data) look
like the EA's time-stamped survey archive (surveys flown all year, 500 m and
2 km tiles) rather than the National LIDAR Programme, so the matching target
is that survey's own DTM. --product auto (default) takes, for the survey
year, lidar_tiles_dtm (the time-stamped DTM of the same surveys) at the
finest resolution up to 1 m, else national_lidar_programme_dtm. Name one to
force it; lidar_composite_dtm (mixed-year mosaic) only if nothing else
exists. preprocess's quality gate rejects scenes whose DTM does not match
their points.

    python -m groundiff.data.ea_dtm --tiles data/laz --out data/ea_dtm
    python -m groundiff.data.ea_dtm --tiles data/laz --out data/ea_dtm --dry-run   # list, no download

The survey year is read from the point tile names (e.g. _20220315_ -> 2022);
override with --year. Tiles already extracted are skipped, so the command can
be re-run after an interruption.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .osgrid import LETTERS, parse_tile

SEARCH_URL = "https://environment.data.gov.uk/backend/catalog/api/tiles/collections/survey/search"
TILE_URL = "https://environment.data.gov.uk/tiles/collections/survey/{product}/{year}/{res}/{tile}"
KEY = "dspui"
HEADERS = {"Content-Type": "application/geo+json", "Accept": "*/*",
           "Origin": "https://environment.data.gov.uk", "Referer": "https://environment.data.gov.uk/survey",
           "User-Agent": "groundiff-ea-dtm/1.0"}
DATE_RE = re.compile(r"_(20\d{2})[01]\d[0-3]\d(?:_|\.|$)")


def grid_letters(e: float, n: float) -> str:
    e100, n100 = int(e // 100_000), int(n // 100_000)
    l1 = (19 - n100) - (19 - n100) % 5 + (e100 + 10) // 5
    l2 = ((19 - n100) * 5) % 25 + e100 % 5
    return LETTERS[l1] + LETTERS[l2]


def tile5k(e: float, n: float) -> dict:
    """5 km survey tile containing (e, n): id 'TL4075', label 'TL47nw', bounds."""
    x0, y0 = int(e // 5000) * 5000, int(n // 5000) * 5000
    sq = grid_letters(x0 + 1, y0 + 1)
    ek, nk = (x0 % 100_000) // 1000, (y0 % 100_000) // 1000
    quad = {(0, 0): "sw", (1, 0): "se", (0, 1): "nw", (1, 1): "ne"}[((ek % 10) // 5, (nk % 10) // 5)]
    return {"id": f"{sq}{ek:02d}{nk:02d}", "label": f"{sq}{ek // 10}{nk // 10}{quad}",
            "bounds": (x0, y0, x0 + 5000, y0 + 5000)}


def tiles_for(bounds) -> list[dict]:
    """All 5 km tiles touching a box (a tile on a 5 km boundary needs both sides)."""
    x0, y0, x1, y1 = bounds
    out = {}
    for x in range(int(x0 // 5000) * 5000, int(x1), 5000):
        for y in range(int(y0 // 5000) * 5000, int(y1), 5000):
            t = tile5k(x + 1, y + 1)
            out[t["id"]] = t
    return list(out.values())


def survey_year(name: str) -> str | None:
    m = DATE_RE.search(Path(name).name)
    return m.group(1) if m else None


def wanted_tiles(point_files: list, year: str | None = None) -> dict[tuple[str, str], dict]:
    """{(tile id, year): tile} for the 5 km DTM tiles covering the point tiles."""
    want = {}
    for p in point_files:
        t = parse_tile(Path(p).name)
        if Path(p).exists():              # the file's own bounds (4-digit names may be 1 or 2 km)
            from .laz import header_bounds
            b = header_bounds(p)
        elif t:
            b = t["extent"]
        else:
            raise ValueError(f"{p}: not found and no OS tile name")
        y = year or survey_year(Path(p).name)
        if not y:
            raise ValueError(f"{Path(p).name}: no survey date in the name; pass --year")
        for t5 in tiles_for(b):
            want[(t5["id"], y)] = t5
    return want


def _lonlat(xs, ys):
    """BNG -> WGS84 with the PROJ that rasterio (or GDAL) bundles; no pyproj needed."""
    try:
        from rasterio.warp import transform
        return transform("EPSG:27700", "EPSG:4326", xs, ys)
    except ImportError:
        from osgeo import osr
        a, b = osr.SpatialReference(), osr.SpatialReference()
        a.ImportFromEPSG(27700)
        b.ImportFromEPSG(4326)
        b.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        a.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(a, b)
        pts = [ct.TransformPoint(x, y)[:2] for x, y in zip(xs, ys)]
        return [p[0] for p in pts], [p[1] for p in pts]


def polygon(bounds, inset: float = 50.0) -> dict:
    x0, y0, x1, y1 = bounds
    xs = [x0 + inset, x1 - inset, x1 - inset, x0 + inset, x0 + inset]
    ys = [y0 + inset, y0 + inset, y1 - inset, y1 - inset, y0 + inset]
    lon, lat = _lonlat(xs, ys)
    return {"type": "Polygon", "coordinates": [[[round(a, 7), round(b, 7)] for a, b in zip(lon, lat)]]}


def search(bounds, timeout: float = 120, attempts: int = 4) -> list[dict]:
    body = json.dumps(polygon(bounds)).encode()
    err = None
    for a in range(attempts):
        try:
            req = urllib.request.Request(SEARCH_URL, data=body, headers=HEADERS, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                res = json.load(r).get("results", [])
            return [{"product": r["product"]["id"], "year": str(r["year"]["id"]), "res": str(r["resolution"]["id"]),
                     "tile": r["tile"]["id"], "label": r["tile"].get("label")} for r in res]
        except urllib.error.HTTPError as e:
            if e.code in (400, 413, 415):
                raise RuntimeError(f"search rejected ({e.code}): {e.read()[:200]!r}") from e
            err = e
        except OSError as e:
            err = e
        time.sleep(2 ** (a + 1))
    raise RuntimeError(f"search failed after {attempts} attempts: {err!r} "
                       "(is environment.data.gov.uk reachable from this machine?)")


def fetch_zip(url: str, dst: Path, attempts: int = 4, timeout: float = 1800) -> Path:
    """The server builds each zip on the fly: no resume, no Content-Length."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    err = None
    for a in range(attempts):
        part = dst.with_suffix(".part")
        part.unlink(missing_ok=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]})
            with urllib.request.urlopen(req, timeout=timeout) as r, part.open("wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            with zipfile.ZipFile(part) as z:
                if z.testzip() is not None:
                    raise zipfile.BadZipFile("CRC error")
            part.replace(dst)
            return dst
        except (OSError, zipfile.BadZipFile) as e:
            err = e
            time.sleep(3 * (a + 1))
    part.unlink(missing_ok=True)
    raise RuntimeError(f"download failed: {url}: {err!r}")


def extract(zpath: Path, out: Path) -> list[Path]:
    """Extract rasters (and the survey metadata GeoPackage) into out/."""
    out.mkdir(parents=True, exist_ok=True)
    got = []
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if name.lower().endswith((".tif", ".tiff", ".tfw", ".gpkg", ".xml")):
                target = out / name
                with z.open(info) as src, open(target, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                if name.lower().endswith((".tif", ".tiff")):
                    got.append(target)
    return got


AUTO_ORDER = ("lidar_tiles_dtm", "national_lidar_programme_dtm")


def _res_m(r: str) -> float:
    """Resolution ids are metres ('1', '2') or labels ('50cm', '0.5')."""
    r = str(r).lower().strip()
    try:
        return float(r[:-2]) / 100 if r.endswith("cm") else float(r.rstrip("m"))
    except ValueError:
        return float("inf")


RES_PREFERENCE = (1.0, 0.5, 2.0)        # 1 m = the training grid; 50 cm averages exactly onto it


def choose(avail: list[dict], tile: str, year: str, product: str = "auto", res: str = "auto") -> dict | None:
    """Pick one catalogue entry for (tile, year): products in AUTO_ORDER (or
    the one named), resolution 1 m, else 50 cm, else 2 m (25 cm tiles are
    ~16x larger than 1 m and are only taken when named with --res)."""
    rows = [r for r in avail if r["tile"] == tile and r["year"] == year]
    products = AUTO_ORDER if product == "auto" else (product,)
    for p in products:
        cand = [r for r in rows if r["product"] == p]
        if res != "auto":
            cand = [r for r in cand if r["res"] == res]
            if cand:
                return cand[0]
            continue
        for want in RES_PREFERENCE:
            hit = [r for r in cand if abs(_res_m(r["res"]) - want) < 1e-6]
            if hit:
                return hit[0]
    return None


def run(point_files: list, out: Path, product: str = "auto", year: str | None = None,
        res: str = "auto", workers: int = 3, dry_run: bool = False, check: bool = True, keep_zip: bool = False,
        log=print) -> dict:
    out = Path(out)
    want = wanted_tiles(point_files, year)
    by_year = defaultdict(list)
    for (tid, y), t in sorted(want.items()):
        by_year[y].append(t)
    log(f"{len(point_files)} point tiles -> {len(want)} DTM tiles of 5 km "
        + ", ".join(f"{y}: {len(v)}" for y, v in sorted(by_year.items())))
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not check and (product == "auto" or res == "auto"):
        raise ValueError("without the catalogue search, name --product and --res")
    done = {(v.get("tile"), v.get("year")) for v in manifest.values()
            if v.get("request") == product and v.get("files") and all(Path(p).exists() for p in v["files"])}
    todo = [(tid, y, t) for (tid, y), t in sorted(want.items()) if (tid, y) not in done]
    log(f"{len(want) - len(todo)} already downloaded, {len(todo)} to fetch (about 50-150 MB each)")
    if dry_run:
        for tid, y, t in todo:
            log(f"  {tid} ({t['label']}) {y}")
        return {"todo": [f"{tid}/{y}" for tid, y, _ in todo], "manifest": manifest}

    def one(item):
        tid, y, t = item
        if check:
            avail = search(t["bounds"])
            pick = choose(avail, tid, y, product, res)
            if pick is None:
                offered = sorted({(r["product"], r["year"], r["res"]) for r in avail if r["tile"] == tid})
                return tid, y, None, f"no {product} DTM for {y} at {tid} ({t['label']}); offered: {offered}"
            prod, rs = pick["product"], pick["res"]
        else:
            prod, rs = product, res
        url = TILE_URL.format(product=prod, year=y, res=rs, tile=tid) + f"?subscription-key={KEY}"
        z = fetch_zip(url, out / "zips" / f"{prod}-{y}-{rs}-{tid}.zip")
        files = extract(z, out / f"{prod}_{y}_{rs}")
        if not keep_zip:
            z.unlink(missing_ok=True)
        if not files:
            return tid, y, None, f"{tid}: the {prod} zip holds no raster (metadata-only delivery)"
        return tid, y, {"product": prod, "res": rs, "files": [str(f) for f in files]}, None

    failed = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(one, item) for item in todo]
        for i, (f, item) in enumerate(zip(futs, todo), 1):
            try:
                tid, y, rec, err = f.result()
            except Exception as e:
                (tid, y, _), rec, err = item, None, repr(e)
            key = f"{tid}/{y}/{product}"
            if err:
                failed[key] = err
                log(f"  [{i}/{len(todo)}] FAILED {err}")
            else:
                manifest[key] = {"tile": tid, "year": y, "request": product, **rec}
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = manifest_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(manifest, indent=1))
                tmp.replace(manifest_path)                     # never a half-written manifest
                log(f"  [{i}/{len(todo)}] {key}: {rec['product']} {rec['res']}: "
                    f"{', '.join(Path(p).name for p in rec['files'])}")
    return {"manifest": manifest, "failed": failed}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", nargs="+", required=True, help="point-cloud tiles or folders (their names give "
                    "the OS tile and survey date)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--product", default="auto",
                    choices=["auto", "lidar_tiles_dtm", "national_lidar_programme_dtm", "lidar_composite_dtm"])
    ap.add_argument("--year", help="survey year (default: from the tile names)")
    ap.add_argument("--res", default="auto", help="resolution id as the catalogue lists it (default: finest <= 1 m)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-check", action="store_true", help="skip the catalogue search, download directly")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    files = []
    for t in a.tiles:
        p = Path(t)
        files += sorted(q for q in p.rglob("*") if q.suffix.lower() in (".laz", ".las")) if p.is_dir() else [p]
    if not files:
        print("no point tiles found", file=sys.stderr)
        return 1
    year = a.year if a.year or a.product != "lidar_composite_dtm" else "2022"
    r = run(files, a.out, a.product, year, a.res, a.workers, a.dry_run, not a.no_check, a.keep_zip)
    if r.get("failed"):
        print(f"{len(r['failed'])} tiles failed; re-run to retry. First: {next(iter(r['failed'].values()))}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

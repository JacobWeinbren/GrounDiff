"""Stratified choice of training tiles from the DEFRA LIDAR 2022 bucket.

A random sample of England is mostly farmland: bridges, quarries,
embankments and flood defences, marsh, towns and coasts - where editors
work hardest - are rare. This scans every tile cheaply and picks a sample
with quotas per landscape type:

  1. every COPC tile is read over HTTP down to the second octree level only
     (~0.4 M of ~25 M points, a couple of seconds): building / high
     vegetation shares, ground height and relief, survey coverage;
  2. OpenStreetMap features per 100 km square (Overpass API): bridges
     (road and railway ways tagged bridge; footpaths left out), quarries, embankments and flood
     defences (man_made=dyke/embankment/flood_wall, embankment=yes,
     wall=flood_wall), marsh (natural=wetland);
  3. each tile gets one stratum (quarry, flood_defence, marsh, coastal,
     urban, suburban, village, woodland, upland, farmland) plus a bridge flag,
     and tiles are drawn to fill the quotas, spread over the 100 km squares.

    python -m groundiff.data.select --out data/v2 --target 700

Writes <out>/selection.json (tiles, strata, features; read by download
--keys-file and preprocess --selection) and <out>/osm.json (bridge and
embankment lines in British National Grid, for preprocess --osm). Scans and
OSM answers are cached under <out>/cache, so a rerun resumes.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .download import BUCKET_URL, list_keys
from .osgrid import parse_tile

STRATA = ("quarry", "flood_defence", "marsh", "coastal", "urban", "suburban", "village", "woodland", "upland",
          "farmland")
# share of the sample per stratum (farmland takes what is left) and for tiles with bridges
QUOTAS = {"quarry": 0.07, "flood_defence": 0.08, "marsh": 0.07, "coastal": 0.07, "urban": 0.10,
          "suburban": 0.08, "village": 0.08, "woodland": 0.08, "upland": 0.08}
BRIDGE_QUOTA = 0.25
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter")

# ----------------------------------------------------------------------------- WGS84 <-> British National Grid
# Ordnance Survey, "A guide to coordinate systems in Great Britain": 7-parameter Helmert
# WGS84 -> OSGB36 (~5 m) and the National Grid transverse Mercator on Airy 1830.
_AIRY = (6377563.396, 6356256.909)
_GRS80 = (6378137.000, 6356752.3141)
_HELMERT = (-446.448, 125.157, -542.060, 20.4894, -0.1502, -0.2470, -0.8421)   # tx ty tz s(ppm) rx ry rz (")
F0, LAT0, LON0, E0, N0 = 0.9996012717, math.radians(49.0), math.radians(-2.0), 400000.0, -100000.0


def _to_cart(lat, lon, ell):
    a, b = ell
    e2 = 1 - b * b / (a * a)
    nu = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return nu * np.cos(lat) * np.cos(lon), nu * np.cos(lat) * np.sin(lon), (1 - e2) * nu * np.sin(lat)


def _from_cart(x, y, z, ell):
    a, b = ell
    e2 = 1 - b * b / (a * a)
    p = np.sqrt(x * x + y * y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(10):
        nu = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2 * nu * np.sin(lat), p)
    return lat, np.arctan2(y, x)


def _helmert(x, y, z, sign=1.0):
    tx, ty, tz, s, rx, ry, rz = (sign * v for v in _HELMERT)
    s = s * 1e-6
    rx, ry, rz = (math.radians(v / 3600.0) for v in (rx, ry, rz))
    return (tx + (1 + s) * x - rz * y + ry * z, ty + rz * x + (1 + s) * y - rx * z,
            tz - ry * x + rx * y + (1 + s) * z)


def _tm(lat, lon):
    a, b = _AIRY
    e2 = 1 - b * b / (a * a)
    n = (a - b) / (a + b)
    s = np.sin(lat)
    nu = a * F0 / np.sqrt(1 - e2 * s * s)
    rho = a * F0 * (1 - e2) / (1 - e2 * s * s) ** 1.5
    eta2 = nu / rho - 1
    dp, sp = lat - LAT0, lat + LAT0
    M = b * F0 * ((1 + n + 1.25 * n ** 2 + 1.25 * n ** 3) * dp
                  - (3 * n + 3 * n ** 2 + 21 / 8 * n ** 3) * np.sin(dp) * np.cos(sp)
                  + (15 / 8 * n ** 2 + 15 / 8 * n ** 3) * np.sin(2 * dp) * np.cos(2 * sp)
                  - 35 / 24 * n ** 3 * np.sin(3 * dp) * np.cos(3 * sp))
    c, t = np.cos(lat), np.tan(lat)
    I_ = M + N0
    II = nu / 2 * s * c
    III = nu / 24 * s * c ** 3 * (5 - t ** 2 + 9 * eta2)
    IIIA = nu / 720 * s * c ** 5 * (61 - 58 * t ** 2 + t ** 4)
    IV = nu * c
    V = nu / 6 * c ** 3 * (nu / rho - t ** 2)
    VI = nu / 120 * c ** 5 * (5 - 18 * t ** 2 + t ** 4 + 14 * eta2 - 58 * t ** 2 * eta2)
    d = lon - LON0
    return (E0 + IV * d + V * d ** 3 + VI * d ** 5,
            I_ + II * d ** 2 + III * d ** 4 + IIIA * d ** 6)


def wgs84_to_bng(lat, lon):
    """Degrees (WGS84) -> (easting, northing) metres, arrays or scalars; ~5 m (Helmert)."""
    lat, lon = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    x, y, z = _helmert(*_to_cart(lat, lon, _GRS80))
    return _tm(*_from_cart(x, y, z, _AIRY))


def bng_to_wgs84(e, n):
    """Inverse of wgs84_to_bng by Newton iteration (for query boxes only)."""
    e, n = float(e), float(n)
    lat, lon = 49.0 + (n + 100000) / 111000.0, -2.0 + (e - 400000) / (111000.0 * math.cos(math.radians(52)))
    for _ in range(20):
        ee, nn = wgs84_to_bng(lat, lon)
        de, dn = e - float(ee), n - float(nn)
        if abs(de) + abs(dn) < 0.01:
            break
        e1, n1 = wgs84_to_bng(lat + 1e-4, lon)
        e2, n2 = wgs84_to_bng(lat, lon + 1e-4)
        J = np.array([[float(e1 - ee), float(e2 - ee)], [float(n1 - nn), float(n2 - nn)]]) / 1e-4
        dlat, dlon = np.linalg.solve(J, [de, dn])
        lat, lon = lat + dlat, lon + dlon
    return lat, lon


# ----------------------------------------------------------------------------- lidar scan

def scan_tile(key: str, levels: int = 2, cell: float = 20.0) -> dict:
    """Coarse statistics of one COPC tile read over HTTP (octree levels < levels)."""
    import laspy
    url = f"{BUCKET_URL}/{urllib.parse.quote(key)}"
    with laspy.CopcReader.open(url) as r:
        h = r.header
        b = (float(h.mins[0]), float(h.mins[1]), float(h.maxs[0]), float(h.maxs[1]))
        p = r.query(level=range(0, levels))
    cls = np.asarray(p.classification)
    x, y, z = np.asarray(p.x), np.asarray(p.y), np.asarray(p.z)
    n = max(len(cls), 1)
    g = cls == 2
    zg = z[g] if g.any() else z
    nx, ny = max(1, int(round((b[2] - b[0]) / cell))), max(1, int(round((b[3] - b[1]) / cell)))
    ix = np.clip(((x - b[0]) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((y - b[1]) / cell).astype(int), 0, ny - 1)
    cover = np.zeros((ny, nx), bool)
    cover[iy, ix] = True
    return {"key": key, "bounds": b, "points": int(h.point_count), "sampled": int(len(cls)),
            "building": float((cls == 6).sum() / n), "high_veg": float((cls == 5).sum() / n),
            "ground": float(g.sum() / n),
            "z_ground_p1": float(np.percentile(zg, 1)) if zg.size else None,
            "z_ground_p50": float(np.percentile(zg, 50)) if zg.size else None,
            "relief": float(np.percentile(zg, 99) - np.percentile(zg, 1)) if zg.size else None,
            "coverage": float(cover.mean())}


def scan_all(keys: list, cache: Path, workers: int = 12, log=print) -> dict:
    """{key: stats}; appended to cache (JSON lines) as they come, so a rerun resumes."""
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            try:
                d = json.loads(line)
                done[d["key"]] = d
            except Exception:
                continue
    todo = [k for k in keys if k not in done]
    log(f"lidar scan: {len(done)} tiles cached, {len(todo)} to read")
    if todo:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "a") as f, ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(_retry, scan_tile, k): k for k in todo}
            for i, fu in enumerate(as_completed(futs), 1):
                k = futs[fu]
                try:
                    d = fu.result()
                except Exception as e:
                    log(f"  [warn] {Path(k).name}: {e!r}")
                    continue
                done[k] = d
                f.write(json.dumps(d) + "\n")
                f.flush()
                if i % 100 == 0 or i == len(todo):
                    log(f"  scanned {i}/{len(todo)}")
    return done


def _retry(fn, *a, tries: int = 4):
    for i in range(tries):
        try:
            return fn(*a)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


# ----------------------------------------------------------------------------- OpenStreetMap

OSM_QUERY = """[out:json][timeout:900];
(
  way["bridge"]["bridge"!="no"]["highway"]["highway"!~"^(footway|path|steps|bridleway|cycleway|corridor)$"]({bbox});
  way["bridge"]["bridge"!="no"]["railway"]({bbox});
  way["man_made"~"^(dyke|embankment|flood_wall)$"]({bbox});
  way["embankment"="yes"]({bbox});
  way["wall"="flood_wall"]({bbox});
);
out tags geom;
(
  way["landuse"="quarry"]({bbox});
  relation["landuse"="quarry"]({bbox});
  way["natural"="wetland"]({bbox});
  relation["natural"="wetland"]({bbox});
);
out tags center;
"""


def osm_kind(tags: dict) -> str | None:
    if tags.get("bridge") not in (None, "no") and ("highway" in tags or "railway" in tags):
        return "bridge"
    if (tags.get("man_made") in ("dyke", "embankment", "flood_wall") or tags.get("embankment") == "yes"
            or tags.get("wall") == "flood_wall"):
        return "flood_defence"
    if tags.get("landuse") == "quarry":
        return "quarry"
    if tags.get("natural") == "wetland":
        return "marsh"
    return None


def parse_overpass(data: dict) -> dict:
    """Overpass JSON -> {"bridge": [lines], "flood_defence": [lines], "quarry": [points], "marsh": [points]}
    in British National Grid (lines as [[e, n], ...])."""
    out = defaultdict(list)
    for el in data.get("elements", []):
        kind = osm_kind(el.get("tags", {}))
        if kind is None:
            continue
        if "geometry" in el:
            lat = [p["lat"] for p in el["geometry"]]
            lon = [p["lon"] for p in el["geometry"]]
            e, n = wgs84_to_bng(lat, lon)
            line = [[round(float(a), 1), round(float(b), 1)] for a, b in zip(np.atleast_1d(e), np.atleast_1d(n))]
            if kind in ("bridge", "flood_defence"):
                out[kind].append(line)
            else:
                out[kind].append([float(np.mean([p[0] for p in line])), float(np.mean([p[1] for p in line]))])
        elif "center" in el:
            e, n = wgs84_to_bng(el["center"]["lat"], el["center"]["lon"])
            out[kind].append([float(e), float(n)])
    return dict(out)


def _overpass(box, tries: int = 6) -> dict:
    """One Overpass query for a (south, west, north, east) box, retrying busy servers."""
    q = OSM_QUERY.replace("{bbox}", "{:.5f},{:.5f},{:.5f},{:.5f}".format(*box))
    err = None
    for attempt in range(tries):
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "groundiff-select/1.0"})
            with urllib.request.urlopen(req, timeout=1200) as r:
                data = json.loads(r.read())
            if "runtime error" in str(data.get("remark", "")):     # out of memory / timeout on the server
                raise RuntimeError(data["remark"])
            return data
        except Exception as e:                   # 429 / 504: the public server is busy
            err = e
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"Overpass failed: {err!r} (rerun later: finished squares are cached)")


def fetch_osm(grids: list, cache_dir: Path, log=print) -> dict:
    """OSM features for the given 100 km squares (letters), one Overpass query each, cached."""
    from .osgrid import os_origin
    cache_dir.mkdir(parents=True, exist_ok=True)
    feats, seen = defaultdict(list), set()
    for gi, grid in enumerate(sorted(grids), 1):
        f = cache_dir / f"osm_{grid}.json"
        if not f.exists():
            e0, n0 = os_origin(grid, 0, 0)
            corners = [bng_to_wgs84(e0 + dx, n0 + dy) for dx in (0, 100000) for dy in (0, 100000)]
            s, w = min(c[0] for c in corners) - 0.02, min(c[1] for c in corners) - 0.02
            nth, est = max(c[0] for c in corners) + 0.02, max(c[1] for c in corners) + 0.02
            box = (s, w, nth, est)
            try:
                data = _overpass(box, tries=3)
            except RuntimeError:                 # too big for the public server: four quarters instead
                mid_lat, mid_lon = (s + nth) / 2, (w + est) / 2
                parts = [_overpass(q, tries=6) for q in ((s, w, mid_lat, mid_lon), (s, mid_lon, mid_lat, est),
                                                         (mid_lat, w, nth, mid_lon), (mid_lat, mid_lon, nth, est))]
                data = {"elements": [e for d in parts for e in d["elements"]]}
            f.write_text(json.dumps(parse_overpass(data)))
            time.sleep(5)                        # be polite to the public server
        got = json.loads(f.read_text())
        for k, v in got.items():               # the query boxes overlap: keep each feature once
            for it in v:
                h = json.dumps(it)
                if h not in seen:
                    seen.add(h)
                    feats[k].append(it)
        log(f"  OSM {grid} ({gi}/{len(grids)}): " + ", ".join(f"{k} {len(v)}" for k, v in sorted(got.items())))
    return dict(feats)


def count_in_tiles(feats: dict, tiles: dict) -> dict:
    """{key: {kind: count}}: lines count when any vertex lies in the tile, points when they do."""
    boxes = {k: t["bounds"] for k, t in tiles.items()}
    cell = 2000.0
    grid = defaultdict(list)                    # coarse spatial index of tiles
    for k, b in boxes.items():
        for gx in range(int(b[0] // cell), int(b[2] // cell) + 1):
            for gy in range(int(b[1] // cell), int(b[3] // cell) + 1):
                grid[(gx, gy)].append(k)
    counts = defaultdict(Counter)
    for kind, items in feats.items():
        for it in items:
            pts = it if isinstance(it[0], list) else [it]
            hit = set()
            for e, n in pts:
                for k in grid.get((int(e // cell), int(n // cell)), ()):
                    b = boxes[k]
                    if b[0] <= e < b[2] and b[1] <= n < b[3]:
                        hit.add(k)
            for k in hit:
                counts[k][kind] += 1
    return counts


# ----------------------------------------------------------------------------- strata and choice

def stratum(t: dict, osm: Counter) -> str:
    if osm.get("quarry"):
        return "quarry"
    if osm.get("flood_defence"):
        return "flood_defence"
    if osm.get("marsh"):
        return "marsh"
    if t["coverage"] < 0.9 and (t.get("z_ground_p1") or 99) < 15:
        return "coastal"
    if t["building"] >= 0.12:
        return "urban"
    if t["building"] >= 0.05:
        return "suburban"
    if t["building"] >= 0.01:
        return "village"
    if t["high_veg"] >= 0.35:
        return "woodland"
    if (t.get("relief") or 0) >= 120 or (t.get("z_ground_p50") or 0) >= 250:
        return "upland"
    return "farmland"


def choose(tiles: dict, target: int, seed: int = 42, bridge_quota: float = BRIDGE_QUOTA,
           quotas: dict | None = None) -> list:
    """Keys, filling the bridge quota first (most bridges first), then each stratum's quota, then the
    rest; within each pool the 100 km square with the fewest tiles so far goes first."""
    quotas = QUOTAS if quotas is None else quotas
    rng = random.Random(seed)
    keys = sorted(tiles)
    rng.shuffle(keys)
    chosen, per_grid, per_stratum = [], Counter(), Counter()

    def take(pool, n, order=None):
        pool = [k for k in pool if k not in chosen_set]
        if order:
            pool.sort(key=order)
        got = 0
        while pool and got < n:
            pool.sort(key=lambda k: per_grid[tiles[k]["grid"]])   # stable: keeps the order within a square
            k = pool.pop(0)
            chosen.append(k)
            chosen_set.add(k)
            per_grid[tiles[k]["grid"]] += 1
            per_stratum[tiles[k]["stratum"]] += 1
            got += 1

    chosen_set = set()
    take([k for k in keys if tiles[k]["osm"].get("bridge")], int(round(bridge_quota * target)),
         order=lambda k: -min(tiles[k]["osm"]["bridge"], 5))
    for s, q in quotas.items():
        need = int(round(q * target)) - per_stratum[s]
        if need > 0:
            take([k for k in keys if tiles[k]["stratum"] == s], need)
    take(keys, target - len(chosen))
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=int, default=700, help="tiles to choose")
    ap.add_argument("--workers", type=int, default=12, help="parallel tile scans")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="folders of tiles to leave out (e.g. earlier test tiles)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    listing = list_keys()
    sizes = dict(listing)
    keys = [k for k, _ in listing if parse_tile(k)]
    skip = {p.name.split(".")[0] for d in a.exclude if d.exists() for p in d.iterdir()}
    keys = [k for k in keys if Path(k).name.split(".")[0] not in skip]
    print(f"{len(keys)} candidate tiles in the DEFRA LIDAR 2022 bucket")
    scans = scan_all(keys, a.out / "cache" / "scan.jsonl", a.workers)
    grids = sorted({parse_tile(k)["grid"] for k in scans})
    print(f"OpenStreetMap features for {len(grids)} 100 km squares")
    feats = fetch_osm(grids, a.out / "cache")
    counts = count_in_tiles(feats, scans)
    tiles = {}
    for k, t in scans.items():
        osm = counts.get(k, Counter())
        tiles[k] = {**t, "grid": parse_tile(k)["grid"], "osm": dict(osm), "stratum": stratum(t, osm)}
    chosen = choose(tiles, a.target, a.seed)
    sel = [{"key": k, "name": Path(k).name, "stratum": tiles[k]["stratum"],
            "bridges": tiles[k]["osm"].get("bridge", 0), "osm": tiles[k]["osm"],
            "stats": {x: tiles[k][x] for x in ("building", "high_veg", "relief", "coverage", "z_ground_p50")}}
           for k in chosen]
    (a.out / "selection.json").write_text(json.dumps({"target": a.target, "tiles": sel}, indent=1))
    (a.out / "osm.json").write_text(json.dumps({"crs": "EPSG:27700", "bridge": feats.get("bridge", []),
                                                "flood_defence": feats.get("flood_defence", [])}))
    all_strata = Counter(t["stratum"] for t in tiles.values())
    got = Counter(s["stratum"] for s in sel)
    print(f"\nchose {len(sel)} tiles ({sum(sizes.get(k, 0) for k in chosen) / 1e9:.1f} GB of LAZ), "
          f"{sum(1 for s in sel if s['bridges'])} with bridges ({sum(s['bridges'] for s in sel)} bridges)")
    print(f"{'stratum':15s} {'chosen':>7s} {'available':>10s}")
    for s in STRATA:
        print(f"{s:15s} {got[s]:7d} {all_strata[s]:10d}")
    print(f"wrote {a.out / 'selection.json'} and {a.out / 'osm.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

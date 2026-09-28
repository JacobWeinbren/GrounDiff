"""Multi-year training data for Noise2Noise: the same places surveyed in several years.

Each EA DTM is one editor's hand-edited reading of that year's lidar. Where the
ground has not changed, the DTMs of different years are independent noisy
labels of the same terrain, so a model trained to predict year b's DTM from
year a's lidar learns the median of the editors' readings (Noise2Noise,
Lehtinen et al., ICML 2018: an L1 loss against independently noisy targets
converges to their median), and a mistake made in one year is outvoted.

    python -m groundiff.data.multiyear plan      --out data/mt --target 60
    python -m groundiff.data.multiyear fetch     --out data/mt
    python -m groundiff.data.multiyear rasterise --out data/mt --workers 3
    python -m groundiff.data.multiyear pairs     --out data/mt

plan       random 5 km OS squares in England (uniformly drawn; the only filter is
           at least --min-years surveys between --years, each with a point cloud
           and a DTM of its own in the EA survey catalogue), and --crops random
           2 km crops of each (one location each). data/mt/plan.json.
fetch      per square and year: the point cloud zip (only the points inside the
           crops, plus a margin, are kept) and the DTM zip; zips are deleted.
build      fetch + rasterise square by square, deleting each square-year's
           points and DTM once its scenes exist (what run_n2n.sh uses: the
           disk then holds the scenes, not every point cloud).
rasterise  per location and year: the usual input rasters and the EA DTM target
           on one 1 m grid shared by all years, with preprocess's quality gate.
           data/mt/scenes/<location>_<year>/.
pairs      per place: its survey years (SAR2SAR pairs); with 3 or more years also
           the consensus DTM (per-cell median of all years' EA DTMs), for
           evaluation only.
compensate SAR2SAR pre-estimates: a trained network's DTM for every scene
           (<name>.npy), used to compensate other years' labels for change.

Every step is resumable (results are cached; rerun after an interruption).
The catalogue and downloads are the EA survey service that ea_dtm uses.
"""
from __future__ import annotations

import argparse
import json
import random
import warnings
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ENGLAND = (82_000, 5_000, 656_000, 658_000)     # bounding box in BNG metres; the catalogue decides what is land
CROP_OFFSETS = ((250, 250), (2750, 250), (250, 2750), (2750, 2750))   # four disjoint 2 km crops in a 5 km square


# ----------------------------------------------------------------------------- catalogue

def family(product: str) -> str:
    """lidar_point_cloud / lidar_tiles_dtm -> lidar; national_lidar_programme_point_cloud / _dtm -> nlp."""
    p = product.lower()
    for suf in ("_point_cloud", "_tiles_dtm", "_dtm"):
        if p.endswith(suf):
            return p[: -len(suf)]
    return p


def is_point_cloud(product: str) -> bool:
    return "point_cloud" in product.lower()


def is_dtm(product: str) -> bool:
    p = product.lower()
    return p.endswith("dtm") and "composite" not in p


def surveys_by_year(avail: list[dict], tile: str, years: range) -> dict:
    """{year: {"pc": offer, "dtm": offer}} for years with a point cloud and a DTM of the same
    product family (the DTM made from that survey), finest DTM resolution first."""
    from .ea_dtm import RES_PREFERENCE, _res_m
    out = {}
    for y in years:
        rows = [r for r in avail if r["tile"] == tile and r["year"] == str(y)]
        pcs = [r for r in rows if is_point_cloud(r["product"])]
        dtms = [r for r in rows if is_dtm(r["product"])]
        best = None
        for pc in sorted(pcs, key=lambda r: r["product"]):
            same = [d for d in dtms if family(d["product"]) == family(pc["product"])]
            same.sort(key=lambda d: min((abs(_res_m(d["res"]) - w), i) for i, w in enumerate(RES_PREFERENCE)))
            same = [d for d in same if _res_m(d["res"]) >= 0.5 - 1e-6]     # no 25 cm (16x the download)
            if same:
                best = {"pc": pc, "dtm": same[0]}
                break
        if best:
            out[str(y)] = best
    return out


def plan(out: Path, target: int, years: range, min_years: int, crops: int, seed: int,
         max_tries: int | None = None, log=print) -> dict:
    """Draw random 5 km squares until `target` have >= min_years surveys; resumable."""
    from .ea_dtm import search, tile5k
    path = out / "plan.json"
    p = json.loads(path.read_text()) if path.exists() else {"squares": [], "tried": [], "seed": seed}
    cache = out / "cache" / "catalogue"
    cache.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    tried = set(p["tried"])
    have = {s["tile"] for s in p["squares"]}
    max_tries = max_tries or 60 * target
    n = 0
    while len(p["squares"]) < target and n < max_tries:
        n += 1
        e, nn = rng.uniform(ENGLAND[0], ENGLAND[2]), rng.uniform(ENGLAND[1], ENGLAND[3])
        t = tile5k(e, nn)
        if t["id"] in tried:
            continue
        tried.add(t["id"])
        cf = cache / f"{t['id']}.json"
        if cf.exists():
            avail = json.loads(cf.read_text())
        else:
            try:
                avail = search(t["bounds"])
            except RuntimeError as err:
                log(f"  {t['id']}: catalogue search failed ({err}); skipped")
                continue
            cf.write_text(json.dumps(avail))
        sv = surveys_by_year(avail, t["id"], years)
        if len(sv) >= min_years and t["id"] not in have:
            offs = rng.sample(CROP_OFFSETS, min(crops, len(CROP_OFFSETS)))
            x0, y0 = t["bounds"][:2]
            p["squares"].append({"tile": t["id"], "label": t["label"], "bounds": t["bounds"], "surveys": sv,
                                 "crops": [[x0 + dx, y0 + dy, x0 + dx + 2000, y0 + dy + 2000] for dx, dy in offs]})
            have.add(t["id"])
            log(f"  [{len(p['squares'])}/{target}] {t['id']} ({t['label']}): years {sorted(sv)}")
        p["tried"] = sorted(tried)
        if n % 20 == 0 or len(p["squares"]) >= target:
            _write_json(path, p)
    _write_json(path, p)
    ys = [len(s["surveys"]) for s in p["squares"]]
    log(f"{len(p['squares'])} squares from {len(tried)} tried; surveys per square: "
        + ", ".join(f"{k} years: {ys.count(k)}" for k in sorted(set(ys))))
    return p


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


# ----------------------------------------------------------------------------- fetch

def _inside(b, boxes, pad: float) -> bool:
    return any(b[0] < x1 + pad and b[2] > x0 - pad and b[1] < y1 + pad and b[3] > y0 - pad
               for x0, y0, x1, y1 in boxes)


COVER_CELL = 100.0          # m: coverage of a crop is the share of its 100 m cells with any point


def clip_laz(src: Path, dst: Path, boxes, pad: float, chunk: int = 2_000_000, cover: list | None = None) -> int:
    """Write the points of src inside any box (+ pad) to dst; returns the number kept.
    cover: one bool array per box ((box size / COVER_CELL)^2), set where a 100 m cell has a point."""
    import laspy
    kept = 0
    with laspy.open(str(src)) as r:
        h = r.header
        if not _inside((h.mins[0], h.mins[1], h.maxs[0], h.maxs[1]), boxes, pad):
            return 0
        tmp = dst.with_suffix(".part.laz")
        nh = laspy.LasHeader(point_format=h.point_format, version=h.version)
        nh.scales, nh.offsets = h.scales, h.offsets
        nh.vlrs.extend(v for v in h.vlrs if getattr(v, "user_id", "").strip("\x00").lower() != "copc")
        with laspy.open(str(tmp), mode="w", header=nh) as w:
            for pts in r.chunk_iterator(chunk):
                x, y = np.asarray(pts.x), np.asarray(pts.y)
                m = np.zeros(x.size, bool)
                for bi, (x0, y0, x1, y1) in enumerate(boxes):
                    m |= (x >= x0 - pad) & (x < x1 + pad) & (y >= y0 - pad) & (y < y1 + pad)
                    if cover is not None:
                        inb = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
                        if inb.any():
                            c = cover[bi]
                            ci = np.minimum(((x[inb] - x0) // COVER_CELL).astype(int), c.shape[1] - 1)
                            ri = np.minimum(((y1 - y[inb]) // COVER_CELL).astype(int), c.shape[0] - 1)
                            c[ri, ci] = True
                if m.any():
                    w.write_points(pts[m])
                    kept += int(m.sum())
    if kept:
        tmp.replace(dst)
    else:
        tmp.unlink(missing_ok=True)
    return kept


def candidate_boxes(sq: dict) -> list:
    """The four disjoint 2 km crops of a 5 km square."""
    x0, y0 = sq["bounds"][:2]
    return [[x0 + dx, y0 + dy, x0 + dx + 2000, y0 + dy + 2000] for dx, dy in CROP_OFFSETS]


def is_empty(out: Path, tile: str, year: str) -> bool:
    return (out / "laz" / tile / year / "EMPTY").exists()


def fetch_one(out: Path, sq: dict, year: str, pad: float = 30.0, keep_zip: bool = False,
              boxes: list | None = None) -> dict:
    """Point cloud (clipped to boxes, default the square's crops) and DTM of one square and year.
    Writes coverage.json (each box's share of 100 m cells with points). A survey with no points in
    any box is marked EMPTY (its DTM is not downloaded) so it is not downloaded again."""
    boxes = boxes if boxes is not None else sq["crops"]
    from .ea_dtm import KEY, TILE_URL, extract, fetch_zip
    s = sq["surveys"][year]
    tile = sq["tile"]
    laz_dir = out / "laz" / tile / year
    dtm_dir = out / "dtm" / tile / year
    rec = {"tile": tile, "year": year}
    if is_empty(out, tile, year):
        return {**rec, "empty": True}
    if not (laz_dir / "DONE").exists():
        pc = s["pc"]
        url = TILE_URL.format(product=pc["product"], year=year, res=pc["res"], tile=tile) + f"?subscription-key={KEY}"
        z = fetch_zip(url, out / "zips" / f"pc-{pc['product']}-{year}-{tile}.zip")
        laz_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        side = int(round((boxes[0][2] - boxes[0][0]) / COVER_CELL))
        cover = [np.zeros((side, side), bool) for _ in boxes]
        with zipfile.ZipFile(z) as zf, tempfile.TemporaryDirectory(dir=out) as td:
            for info in zf.infolist():
                name = Path(info.filename).name
                if not name.lower().endswith((".laz", ".las")) or ".." in Path(info.filename).parts:
                    continue
                tmp = Path(td) / name
                with zf.open(info) as a, open(tmp, "wb") as b:
                    while chunk := a.read(1 << 20):
                        b.write(chunk)
                dst = laz_dir / (Path(name).stem.replace(".copc", "") + ".laz")
                n += clip_laz(tmp, dst, boxes, pad, cover=cover)
                tmp.unlink(missing_ok=True)
        if not keep_zip:
            z.unlink(missing_ok=True)
        if n == 0:
            (laz_dir / "EMPTY").write_text("no points inside the crops in the point-cloud zip")
            return {**rec, "empty": True}
        _write_json(laz_dir / "coverage.json", {"boxes": boxes, "cover": [float(c.mean()) for c in cover]})
        (laz_dir / "DONE").write_text(str(n))
        rec["points"] = n
    if not (dtm_dir / "DONE").exists():
        d = s["dtm"]
        url = TILE_URL.format(product=d["product"], year=year, res=d["res"], tile=tile) + f"?subscription-key={KEY}"
        z = fetch_zip(url, out / "zips" / f"dtm-{d['product']}-{year}-{tile}.zip")
        files = extract(z, dtm_dir)
        if not keep_zip:
            z.unlink(missing_ok=True)
        if not files:
            raise RuntimeError(f"{tile} {year}: the DTM zip holds no raster")
        (dtm_dir / "DONE").write_text(str(len(files)))
    return rec


def fetch(out: Path, workers: int = 2, log=print) -> dict:
    p = json.loads((out / "plan.json").read_text())
    jobs = [(sq, y) for sq in p["squares"] for y in sorted(sq["surveys"])]
    todo = [(sq, y) for sq, y in jobs if not is_empty(out, sq["tile"], y)
            and not ((out / "laz" / sq["tile"] / y / "DONE").exists() and (out / "dtm" / sq["tile"] / y / "DONE").exists())]
    log(f"{len(jobs)} square-years, {len(jobs) - len(todo)} done, {len(todo)} to download")
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(fetch_one, out, sq, y): (sq["tile"], y) for sq, y in todo}
        for i, f in enumerate(as_completed(futs), 1):
            tile, y = futs[f]
            try:
                f.result()
                log(f"  [{i}/{len(todo)}] {tile} {y}")
            except Exception as e:
                failed.append(f"{tile} {y}: {e}")
                log(f"  [{i}/{len(todo)}] FAILED {tile} {y}: {e}")
    return {"failed": failed}


# ----------------------------------------------------------------------------- rasterise

def location_id(tile: str, crop) -> str:
    return f"{tile}_{int(crop[0]) % 5000:04d}_{int(crop[1]) % 5000:04d}"


def rasterise_one(out: Path, sq: dict, crop, year: str, gsd: float = 1.0, gate: dict | None = None,
                  overwrite: bool = False) -> dict | None:
    from .preprocess import _default_crs, _save, index_rasters, is_suspect, quality, rasters_for
    from .rasterise import Grid, target_from_rasters
    from .stream import Accumulator, stream_file
    loc = location_id(sq["tile"], crop)
    name = f"{loc}_{year}"
    sd = out / "scenes" / name
    if (sd / "meta.json").exists() and not overwrite:
        return None
    files = sorted((out / "laz" / sq["tile"] / year).glob("*.laz"))
    if not files:
        raise RuntimeError(f"{name}: no point files")
    x0, y0, x1, y1 = crop
    grid = Grid(float(x0), float(y1), gsd, int(round((x1 - x0) / gsd)), int(round((y1 - y0) / gsd)))
    acc = Accumulator(grid, keep_ground=True, ground_classes=(2,), ground_per_cell=True)
    for f in files:
        stream_file(f, grid.bounds, acc, {"drop_withheld": True})
    if acc.n_points == 0:
        raise RuntimeError(f"{name}: no points inside the crop")
    hist = acc.class_histogram()
    arrs = acc.rasters(lasground=True)
    ground_tin = arrs.pop("dtm_before", None)             # published class 2: the quality check only
    for k in ("before_valid", "sem_ground", "sem_nonground"):
        arrs.pop(k, None)
    survey = arrs["in_survey"] > 0.5
    index = index_rasters(out / "dtm" / sq["tile"] / year)
    paths = rasters_for(grid.bounds, index, year, files[0].name)
    if not paths:
        raise RuntimeError(f"{name}: no DTM raster covers the crop")
    arrs.update(target_from_rasters(grid, paths, survey))
    q = quality(arrs, ground_tin=ground_tin)
    why = is_suspect(q, **(gate or {}))
    meta = {"schema": 3, "scene": name, "location": loc, "year": year, "tile": sq["tile"], "gsd": gsd,
            "grid": grid.to_dict(), "crs_wkt": acc.crs_wkt or _default_crs(), "n_points": acc.n_points,
            "class_hist_before": hist, "has_before": False, "lasground": False, "read_opts": {"drop_withheld": True},
            "target": "dtm_raster", "dtm_files": paths, "point_files": [str(f) for f in files],
            "surveys": sq["surveys"][year], "quality": {**q, "suspect": bool(why), "reasons": why}}
    sd.mkdir(parents=True, exist_ok=True)
    for k, v in arrs.items():
        _save(sd, k, v)
    _write_json(sd / "meta.json", meta)
    return meta


def _rasterise_job(args):
    out, sq, crop, year, gate = args
    try:
        m = rasterise_one(out, sq, crop, year, gate=gate)
        return location_id(sq["tile"], crop), year, m, None
    except RuntimeError as e:                             # no points / no DTM here: will not change
        _skip(out, f"{location_id(sq['tile'], crop)}_{year}", str(e))
        return location_id(sq["tile"], crop), year, None, repr(e)
    except Exception as e:
        return location_id(sq["tile"], crop), year, None, repr(e)


def _skip(out: Path, scene: str, why: str) -> None:
    d = out / "skipped"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{scene}.txt").write_text(why)


def _scene_settled(out: Path, scene: str) -> bool:
    """Rasterised, or known not to be possible (skipped/<scene>.txt)."""
    return (out / "scenes" / scene / "meta.json").exists() or (out / "skipped" / f"{scene}.txt").exists()


def _isolated(job):
    """_rasterise_job in a process of its own: a worker killed for memory loses only its scene."""
    from concurrent.futures.process import BrokenProcessPool
    try:
        with ProcessPoolExecutor(max_workers=1) as px:
            return px.submit(_rasterise_job, job).result()
    except BrokenProcessPool:
        _, sq, crop, year, _ = job
        return (location_id(sq["tile"], crop), year, None,
                "the rasterising process was killed (probably out of memory; retried on the next run)")


def dtm_coverage(out: Path, sq: dict, year: str, boxes: list) -> list:
    """Each box's share of cells with an EA DTM value (sampled on a 10 m grid)."""
    from .preprocess import index_rasters, rasters_for
    from .rasterise import Grid, target_from_rasters
    index = index_rasters(out / "dtm" / sq["tile"] / year)
    res = []
    for x0, y0, x1, y1 in boxes:
        g = Grid(float(x0), float(y1), 10.0, int((x1 - x0) // 10), int((y1 - y0) // 10))
        paths = rasters_for(g.bounds, index, year)
        if not paths:
            res.append(0.0)
            continue
        t = target_from_rasters(g, paths, np.ones((g.height, g.width), bool))
        res.append(float(np.mean(np.isfinite(t["gt_dtm"]) & (t["gt_valid"] > 0.5))))
    return res


def choose_crops(out: Path, sq: dict, n: int) -> tuple[list, list]:
    """The n candidate crops with the most coverage summed over the years: per year min(share of
    100 m cells with points, share of cells with a DTM value). Crops with no coverage are dropped."""
    boxes = candidate_boxes(sq)
    score = np.zeros(len(boxes))
    for y in sorted(sq["surveys"]):
        cf = out / "laz" / sq["tile"] / y / "coverage.json"
        if is_empty(out, sq["tile"], y) or not cf.exists():
            continue
        pc = json.loads(cf.read_text())["cover"]
        dt = dtm_coverage(out, sq, y, boxes)
        score += np.minimum(pc, dt)
    order = [int(i) for i in np.argsort(-score, kind="stable") if score[i] > 0][:n]
    return [boxes[i] for i in order], [float(score[i]) for i in order]


def rasterise(out: Path, workers: int = 3, gate: dict | None = None, log=print) -> dict:
    p = json.loads((out / "plan.json").read_text())
    jobs = [(out, sq, crop, y, gate) for sq in p["squares"] for crop in sq["crops"] for y in sorted(sq["surveys"])
            if (out / "laz" / sq["tile"] / y / "DONE").exists() and (out / "dtm" / sq["tile"] / y / "DONE").exists()]
    log(f"{len(jobs)} location-years to rasterise")
    failed, suspect = [], 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, (loc, y, m, err) in enumerate(ex.map(_rasterise_job, jobs), 1):
            if err:
                failed.append(f"{loc} {y}: {err}")
                log(f"  [{i}/{len(jobs)}] FAILED {loc} {y}: {err}")
                continue
            if m is not None:
                suspect += bool(m["quality"]["suspect"])
                log(f"  [{i}/{len(jobs)}] {loc} {y}" + (f"  SUSPECT: {'; '.join(m['quality']['reasons'])}"
                                                         if m["quality"]["suspect"] else ""))
    log(f"done: {len(jobs) - len(failed)} scenes ({suspect} new suspect ones, skipped by training), {len(failed)} failed")
    return {"failed": failed}


def _prune(out: Path, sq: dict, year: str) -> bool:
    """Delete one square-year's clipped points and DTM once every crop's scene exists (DONE markers
    stay, so nothing is downloaded again); returns whether it did."""
    if not all(_scene_settled(out, f"{location_id(sq['tile'], c)}_{year}") for c in sq["crops"]):
        return False
    for d in (out / "laz" / sq["tile"] / year, out / "dtm" / sq["tile"] / year):
        if d.is_dir():
            for f in d.rglob("*"):
                if f.is_file() and f.name not in ("DONE", "EMPTY", "coverage.json"):
                    f.unlink()
    return True


def build(out: Path, fetch_workers: int = 2, workers: int = 2, gate: dict | None = None, log=print) -> dict:
    """fetch + rasterise square by square, deleting each square-year's points and DTM once its scenes
    are settled: the disk holds the scenes plus the squares in flight, not every point cloud.
    A square not started yet is clipped to all four candidate crops for every year first; then the
    crops the surveys actually cover (choose_crops) replace the plan's random ones. Squares already
    started keep their crops. Each scene is rasterised in its own process, at most `workers` at once."""
    import threading
    path = out / "plan.json"
    p = json.loads(path.read_text())
    squares = p["squares"]
    lock, slots = threading.Lock(), threading.Semaphore(max(1, workers))
    failed, done = [], [0]
    n_years = sum(len(sq["surveys"]) for sq in squares)

    def rasterise_job(job):
        with slots:
            return _isolated(job)

    def one_square(sq):
        errs = []
        tile, years = sq["tile"], sorted(sq["surveys"])
        legacy = any((out / "laz" / tile / y / "DONE").exists() and not (out / "laz" / tile / y / "coverage.json").exists()
                     for y in years) or any((out / "scenes").glob(f"{tile}_*"))
        if not sq.get("crops_chosen") and not legacy:
            n = len(sq["crops"])
            for y in years:
                try:
                    fetch_one(out, sq, y, boxes=candidate_boxes(sq))
                except Exception as e:
                    errs.append(f"{tile} {y}: {e}")
            crops, scores = choose_crops(out, sq, n)
            with lock:
                sq["crops"], sq["crop_scores"], sq["crops_chosen"] = crops, scores, True
                _write_json(path, p)
            log(f"  {tile}: crops " + (", ".join(f"{location_id(tile, c)} (coverage {sc:.1f} survey-years)"
                                                for c, sc in zip(crops, scores)) or "none covered"))
        for y in years:
            before = len(errs)
            names = [f"{location_id(tile, c)}_{y}" for c in sq["crops"]]
            try:
                if not all(_scene_settled(out, nm) for nm in names):
                    r = fetch_one(out, sq, y)
                    if r.get("empty"):
                        for nm in names:
                            _skip(out, nm, "no points in the survey's point cloud here")
                    else:
                        jobs = [(out, sq, c, y, gate) for c, nm in zip(sq["crops"], names)
                                if not _scene_settled(out, nm)]
                        with ThreadPoolExecutor(max_workers=len(jobs) or 1) as tx:
                            for loc, _, m, err in tx.map(rasterise_job, jobs):
                                if err:
                                    errs.append(f"{loc} {y}: {err}")
                                elif m is not None and m["quality"]["suspect"]:
                                    log(f"    {loc} {y} SUSPECT: {'; '.join(m['quality']['reasons'])}")
                if all(_scene_settled(out, nm) for nm in names):
                    _prune(out, sq, y)
            except Exception as e:
                errs.append(f"{tile} {y}: {e}")
            with lock:
                done[0] += 1
                log(f"  [{done[0]}/{n_years}] {tile} {y}" + "".join(f"\n      failed: {e}" for e in errs[before:]))
        return errs

    with ThreadPoolExecutor(max_workers=max(1, fetch_workers)) as tx:
        for errs in tx.map(one_square, squares):
            failed += errs
    n = sum(1 for _ in (out / "scenes").glob("*/meta.json")) if (out / "scenes").exists() else 0
    n_skip = sum(1 for _ in (out / "skipped").glob("*.txt")) if (out / "skipped").exists() else 0
    log(f"done: {n} scenes; {n_skip} place-years not covered by their survey (data/mt/skipped/); "
        f"{len(failed)} failures this run")
    return {"failed": failed}


# ----------------------------------------------------------------------------- pairs, consensus

def pairs(out: Path, min_consensus: int = 3, log=print) -> dict:
    """For every place: which survey years it has (the SAR2SAR pairs) and, with min_consensus or more
    years, the consensus DTM: the per-cell median of all years' EA DTMs, for evaluation only (the value
    an L1 Noise2Noise model converges to where the ground did not change)."""
    from .preprocess import _save
    root = out / "scenes"
    locs: dict[str, list] = {}
    for m in sorted(root.glob("*/meta.json")):
        meta = json.loads(m.read_text())
        if meta.get("quality", {}).get("suspect") or "location" not in meta:
            continue
        locs.setdefault(meta["location"], []).append(meta)
    stats = {"locations": len(locs), "pairs": 0, "consensus": 0}
    for loc, metas in sorted(locs.items()):
        metas.sort(key=lambda m: m["year"])
        years = [m["year"] for m in metas]
        labels = {}
        for m in metas:
            sd = root / m["scene"]
            gt = np.load(sd / "gt_dtm.npy").astype(np.float32)
            labels[m["year"]] = np.where(np.load(sd / "gt_valid.npy") > 0.5, gt, np.nan)
        stats["pairs"] += len(years) * (len(years) - 1) // 2
        cons = None
        if len(years) >= min_consensus:
            S = np.stack([labels[y] for y in years])
            n = np.isfinite(S).sum(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)          # cells no year covers
                cons = np.nanmedian(np.where(n >= min_consensus, S, np.nan), axis=0)
            cons = np.where(n >= min_consensus, cons, np.nan).astype(np.float32)
        for m in metas:
            a = m["year"]
            sd = root / m["scene"]
            for f in list(sd.glob("unchanged_*.npy")) + [sd / "label_above_lidar.npy", sd / "gt_consensus.npy",
                                                         sd / "label_error.npy"]:
                f.unlink(missing_ok=True)                  # from earlier versions of this step
            for k in ("label_noise", "label_above_lidar_frac"):
                m.pop(k, None)
            m["pairs"] = {b: {"scene": f"{loc}_{b}"} for b in years if b != a}
            if cons is not None:
                err = (labels[a] - cons).astype(np.float32)
                _save(sd, "gt_consensus", cons)
                _save(sd, "label_error", err)
                ok = np.isfinite(err)
                if ok.any():
                    e = np.abs(err[ok])
                    m["label_noise"] = {"cells": int(ok.sum()), "rmse_vs_consensus": float(np.sqrt(np.mean(e ** 2))),
                                        "frac_over_0.2m": float((e > 0.2).mean()), "frac_over_0.5m": float((e > 0.5).mean())}
                stats["consensus"] += 1
            _write_json(sd / "meta.json", m)
        log(f"  {loc}: years {years}")
    log(f"{stats['locations']} places, {stats['pairs']} year pairs, {stats['consensus']} scenes with a consensus DTM")
    return stats


def compensate(out: Path, checkpoint: str, name: str, device: str = "auto", batch_size: int = 8,
               log=print) -> int:
    """SAR2SAR's pre-estimates (Dalsasso et al. 2021, Sec. IV-B and V-B): the given network's DTM for
    every non-suspect scene of a place with two or more years, saved as <name>.npy (metres), used to
    compensate the other years' labels for change (dataset.PairScene)."""
    from ..infer import load_net
    from ..runtime import predict_scene
    net, spec = load_net(checkpoint, None, device)
    root = out / "scenes"
    todo = []
    for m in sorted(root.glob("*/meta.json")):
        meta = json.loads(m.read_text())
        if meta.get("pairs") and not meta.get("quality", {}).get("suspect"):
            todo.append(m.parent)
    log(f"{len(todo)} scenes: pre-estimates '{name}' from {checkpoint}")
    for i, sd in enumerate(todo, 1):
        meta = json.loads((sd / "meta.json").read_text())
        if meta.get("compensation", {}).get(name) == str(checkpoint) and (sd / f"{name}.npy").exists():
            continue                                        # resumable
        arrs = {n: np.load(sd / f"{n}.npy").astype(np.float32) for n in spec.needed_channels}
        for n in ("in_survey", "has_return"):
            if (sd / f"{n}.npy").exists():
                arrs[n] = np.load(sd / f"{n}.npy").astype(np.float32)
        res = predict_scene(arrs, spec, net, batch_size=batch_size, gsd=meta["grid"]["gsd"])
        np.save(sd / f"{name}.npy", res["dtm"].astype(np.float32))
        meta.setdefault("compensation", {})[name] = str(checkpoint)
        _write_json(sd / "meta.json", meta)
        log(f"  [{i}/{len(todo)}] {sd.name}")
    return len(todo)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("plan", "fetch", "rasterise", "build", "pairs", "compensate"):
        s = sub.add_parser(c)
        s.add_argument("--out", type=Path, required=True)
        if c == "plan":
            s.add_argument("--target", type=int, default=60, help="5 km squares")
            s.add_argument("--years", type=int, nargs=2, default=[2017, 2022])
            s.add_argument("--min-years", type=int, default=2)
            s.add_argument("--crops", type=int, default=2, help="2 km crops (locations) per square, at most 4")
            s.add_argument("--seed", type=int, default=42)
        if c in ("fetch", "rasterise", "build"):
            s.add_argument("--workers", type=int, default=2,
                           help="download threads (fetch) or rasterising processes")
        if c == "build":
            s.add_argument("--fetch-workers", type=int, default=2, help="squares downloaded at once")
        if c == "compensate":
            s.add_argument("--checkpoint", required=True)
            s.add_argument("--name", required=True, help="e.g. xhat_a (from step A) or xhat_b (from step B)")
            s.add_argument("--device", default="auto")
            s.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "plan":
        plan(a.out, a.target, range(a.years[0], a.years[1] + 1), a.min_years, a.crops, a.seed)
    elif a.cmd == "fetch":
        r = fetch(a.out, a.workers)
        return 1 if r["failed"] else 0
    elif a.cmd == "rasterise":
        r = rasterise(a.out, a.workers)
        return 1 if r["failed"] else 0
    elif a.cmd == "build":
        r = build(a.out, a.fetch_workers, a.workers)
        return 1 if r["failed"] else 0
    elif a.cmd == "compensate":
        compensate(a.out, a.checkpoint, a.name, a.device, a.batch_size)
    else:
        pairs(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

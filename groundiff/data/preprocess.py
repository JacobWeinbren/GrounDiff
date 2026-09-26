"""Rasterise scenes into per-channel .npy files (memory-mappable).

One scene = one EA point-cloud tile ("after": the final, hand-edited
classification) and optionally the same points re-classified by
lasground_new ("before"). Output layout:

    <out>/<scene>/meta.json
    <out>/<scene>/<channel>.npy          float32 [H, W], NaN = no data

Channels: dsm_max, dsm_min, dsm_last, density, z_std, echoes, has_return,
gt_dtm (+ gt_valid), top_ground (highest return is a target-ground class), and with a "before" file also dtm_before
(+ before_valid), sem_ground, sem_nonground (DeepTerRa's 2-channel semantic
raster built from the lasground_new classes).

Usage:
    python -m groundiff.data.preprocess --after-dir EA_LAZ/ --before-dir LASGROUND_LAZ/ \
        --out data/scenes_05m --gsd 0.5
Scenes are matched by file stem (".copc" is stripped).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .laz import NOISE_CLASSES, class_histogram, read_points
from .rasterise import Grid, build_rasters

SCHEMA = 2
LAZ_SUFFIXES = (".laz", ".las")


def scene_name(path: Path) -> str:
    name = path.name
    for suf in (".copc.laz", ".laz", ".las"):
        if name.lower().endswith(suf):
            return name[: -len(suf)]
    return path.stem


def _save(out: Path, name: str, arr: np.ndarray):
    np.save(out / f"{name}.npy", np.ascontiguousarray(arr, dtype=np.float32))


def process_scene(after: Path, out_root: Path, before: Path | None = None, gsd: float = 0.5,
                  ground_classes=(2, 9), before_ground_classes=(2,),
                  overwrite: bool = False, read_opts: dict | None = None) -> dict | None:
    """read_opts: drop_overlap / drop_synthetic / drop_withheld for read_points."""
    read_opts = read_opts or {}
    name = scene_name(after)
    out = out_root / name
    meta_path = out / "meta.json"
    if meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text())
        if (meta.get("schema") == SCHEMA and meta.get("gsd") == gsd
                and meta.get("has_before") == (before is not None)):
            return None
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pts = read_points(after, drop_classes=NOISE_CLASSES, **read_opts)
    grid = Grid.from_bounds(pts.x.min(), pts.y.min(), pts.x.max(), pts.y.max(), gsd)
    # The before file is used as lasground_new wrote it (only its own noise
    # classes are dropped) so no EA editing leaks into the inputs.
    bp = read_points(before, drop_classes=NOISE_CLASSES, **read_opts) if before is not None else None
    for k, v in build_rasters(grid, pts, bp, ground_classes, before_ground_classes).items():
        _save(out, k, v)
    meta = {
        "schema": SCHEMA, "scene": name, "gsd": gsd, "grid": grid.to_dict(),
        "crs_wkt": pts.crs_wkt, "after_file": str(after), "n_points": len(pts),
        "ground_classes": list(ground_classes), "class_hist_after": class_histogram(pts.cls),
        "has_before": before is not None, "read_opts": read_opts,
    }
    if bp is not None:
        meta.update({"before_file": str(before), "n_points_before": len(bp),
                     "before_ground_classes": list(before_ground_classes),
                     "class_hist_before": class_histogram(bp.cls)})
        if len(bp) != len(pts):
            meta["warning"] = ("before/after point counts differ after noise removal "
                               f"({len(bp)} vs {len(pts)})")
    meta["seconds"] = round(time.time() - t0, 1)
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


def find_laz(folder: Path) -> dict[str, Path]:
    files = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in LAZ_SUFFIXES]
    return {scene_name(p): p for p in files}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--after-dir", type=Path, required=True, help="EA classified LAZ/COPC tiles")
    ap.add_argument("--before-dir", type=Path, help="same tiles re-classified by lasground_new")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gsd", type=float, default=0.5)
    ap.add_argument("--ground-classes", type=int, nargs="+", default=[2, 9],
                    help="classes forming the target DTM (default: ground + water)")
    ap.add_argument("--before-ground-classes", type=int, nargs="+", default=[2])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--drop-overlap", action="store_true", help="drop overlap-flagged / class 12 points")
    ap.add_argument("--drop-synthetic", action="store_true", help="drop synthetic-flagged points")
    ap.add_argument("--keep-withheld", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    after = find_laz(a.after_dir)
    before = find_laz(a.before_dir) if a.before_dir else {}
    if a.before_dir:
        missing = sorted(set(after) - set(before))
        if missing:
            print(f"[warn] {len(missing)} scenes have no before file and are skipped: {missing[:5]}...")
        after = {k: v for k, v in after.items() if k in before}
    a.out.mkdir(parents=True, exist_ok=True)
    ro = {"drop_overlap": a.drop_overlap, "drop_synthetic": a.drop_synthetic,
          "drop_withheld": not a.keep_withheld}
    jobs = [(p, a.out, before.get(k), a.gsd, tuple(a.ground_classes),
             tuple(a.before_ground_classes), a.overwrite, ro) for k, p in after.items()]
    print(f"{len(jobs)} scenes -> {a.out}")
    failures = 0
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futures = {ex.submit(process_scene, *j): j[0] for j in jobs}
        for fut in futures:
            try:
                meta = fut.result()
                print(f"  {futures[fut].name}: " + ("cached" if meta is None else f"{meta['seconds']} s"))
            except Exception as e:   # keep going; report at the end
                failures += 1
                print(f"  {futures[fut].name}: FAILED {e!r}", file=sys.stderr)
    if failures:
        print(f"{failures} scenes failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

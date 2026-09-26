"""Predict DTMs for whole scenes, write GeoTIFFs, rank areas for editing.

    python -m groundiff.infer --checkpoint runs/before_after/best.pt \
        --scenes data/scenes_05m --split-file data/split.json --split test --out results/test
    python -m groundiff.infer --onnx models/before_after.onnx --scenes ... --out ...

Outputs per scene (GeoTIFF, same grid as the inputs):
    dtm.tif          predicted DTM (m)
    p_ground.tif     GrounDiff confidence sigmoid(l) (0-1) that the gate surface is right
                     (DSM-gated models only)
    p_edit.tif       1 - sigmoid(l) when the gate is the lasground_new DTM: edit probability
    std.tif          spread across samples / TTA views (m), with --samples > 1 or --tta
    dz_before.tif    predicted DTM - lasground_new DTM (m): the predicted edit
    error.tif        predicted DTM - reference DTM (m), when a reference exists
plus coloured overlays for LP360 (see overlay.py): p_edit_overlay.tif (RGBA),
p_edit_overlay_rgb.tif (RGB + nodata), likewise for dz_before and std, and
QGIS .qml styles. And, when a reference DTM exists, metrics.json (model and lasground_new vs
reference; GrounDiff's RMSE/MAE/Type I/II/total plus ResDepth's MedAE/NMAD, and
edit precision/recall/F1 against the lasground_new DTM).

priority.csv ranks square blocks (default 100 m, with coordinates) by
predicted edit volume (sum of |dz_before| x cell area). With a reference it also reports the true
edit size and, in summary.json, how much of the true edit volume the top 5,
10 and 20 % of blocks capture. This ranking is our addition; neither paper
defines one.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from .io_raster import write_geotiff
from .metrics import dtm_metrics
from .overlay import write_overlays
from .runtime import RuntimeSpec, predict_scene


def load_net(checkpoint: str | None, onnx: str | None, device: str = "auto"):
    if onnx:
        from .backends import OnnxNet
        spec = RuntimeSpec.from_json(Path(onnx).with_suffix(".json"))
        return OnnxNet(onnx), spec
    from .backends import TorchNet
    from .device import pick_device
    from .models.build import load_model
    dev = pick_device(device)
    model, cfg, ck = load_model(checkpoint, dev)
    return TorchNet(model, dev), RuntimeSpec.from_config(cfg, ck.get("data_meta"))


def overlay_presets(res: dict) -> list[tuple[str, str]]:
    """Which rasters get LP360-ready colour overlays."""
    out = []
    if "p_edit" in res:
        out.append(("p_edit", "edit"))
    if "dz_before" in res:
        out.append(("dz_before", "dz"))
    if "std" in res:
        out.append(("std", "uncertainty"))
    return out


def block_priorities(dz: np.ndarray, gsd: float, block_m: float, std: np.ndarray | None = None,
                     true_dz: np.ndarray | None = None, alpha: float = 0.2,
                     p_edit: np.ndarray | None = None, xmin: float = 0.0, ymax: float = 0.0) -> list[dict]:
    b = max(1, int(round(block_m / gsd)))
    H, W = dz.shape
    rows = []
    for r in range(0, H, b):
        for c in range(0, W, b):
            d = dz[r:r + b, c:c + b]
            ok = np.isfinite(d)
            if not ok.any():
                continue
            ad = np.abs(d[ok])
            hh, ww = d.shape
            row = {"row": r, "col": c, "x_min": xmin + c * gsd, "y_max": ymax - r * gsd,
                   "x_centre": xmin + (c + ww / 2) * gsd, "y_centre": ymax - (r + hh / 2) * gsd,
                   "pred_edit_mean_m": float(ad.mean()), "pred_edit_volume_m3": float(ad.sum()) * gsd * gsd,
                   "pred_edit_frac": float((ad > alpha).mean()), "n": int(ok.sum())}
            if std is not None:
                row["std_mean_m"] = float(np.nanmean(std[r:r + b, c:c + b]))
            if p_edit is not None:
                row["p_edit_mean"] = float(np.nanmean(p_edit[r:r + b, c:c + b]))
            if true_dz is not None:
                td = np.abs(true_dz[r:r + b, c:c + b])
                row["true_edit_mean_m"] = float(np.nanmean(td)) if np.isfinite(td).any() else float("nan")
                row["true_edit_volume"] = float(np.nansum(td)) * gsd * gsd
            rows.append(row)
    rows.sort(key=lambda x: -x["pred_edit_volume_m3"])
    return rows


def capture_at(rows: list[dict], fractions=(0.05, 0.1, 0.2)) -> dict:
    """Share of the true edit volume inside the top-ranked fraction of blocks."""
    total = sum(r.get("true_edit_volume", 0.0) for r in rows)
    out = {}
    for f in fractions:
        k = max(1, int(round(f * len(rows))))
        got = sum(r.get("true_edit_volume", 0.0) for r in rows[:k])
        out[f"capture_top{int(f * 100)}pct"] = got / total if total > 0 else float("nan")
    return out


def _progress(name: str):
    """Print '<scene>: 37% (12 min left)' on one line while a scene runs."""
    t0 = time.time()
    last = [-1]

    def cb(f):
        pct = int(f * 100)
        if pct == last[0]:
            return
        last[0] = pct
        left = (time.time() - t0) * (1 - f) / max(f, 1e-6)
        print(f"\r{name}: {pct:3d}% ({left / 60:.0f} min left)   ", end="" if f < 1 else "\n", flush=True)
    return cb


def run_scene(scene_dir: Path, net, spec: RuntimeSpec, out_dir: Path, args) -> dict:
    meta = json.loads((scene_dir / "meta.json").read_text())
    g = meta["grid"]
    names = set(spec.needed_channels) | {"gt_dtm", "gt_valid", "dsm_max", "in_survey", "has_return"}
    arrs = {n: np.load(scene_dir / f"{n}.npy") for n in names if (scene_dir / f"{n}.npy").exists()}
    res = predict_scene(arrs, spec, net, stride=args.stride, blend=args.blend, prior=args.prior,
                        init=args.init, t_start=args.t_start, n_samples=args.samples, tta=args.tta,
                        batch_size=args.batch_size, seed=args.seed, gsd=g["gsd"],
                        progress=_progress(scene_dir.name))
    out_dir.mkdir(parents=True, exist_ok=True)
    geo = (g["xmin"], g["ymax"], g["gsd"], meta.get("crs_wkt"))
    for k in ("dtm", "p_ground", "p_edit", "std", "dz_before", "prior"):
        if k in res:
            write_geotiff(out_dir / f"{k}.tif", res[k], *geo)
    if not getattr(args, "no_overlays", False):
        for k, preset in overlay_presets(res):
            write_overlays(res[k], out_dir / k, preset, *geo,
                           direction=res.get("dz_before") if k == "p_edit" else None)
    summary = {"scene": meta["scene"]}
    true_dz = None
    if "gt_dtm" in arrs:
        gt = np.where(arrs["gt_valid"] > 0.5, arrs["gt_dtm"], np.nan)
        write_geotiff(out_dir / "error.tif", res["dtm"] - gt, *geo)
        valid = np.isfinite(gt) & np.isfinite(res["dtm"])
        s = arrs["dsm_max"]                   # Type I/II always against the highest-return DSM
        pg = res["p_ground"] > 0.5 if ("p_ground" in res and spec.gate_channel == "dsm_max") else None
        before = arrs.get(spec.prior_channel) if spec.prior_channel else None
        summary["model"] = dtm_metrics(res["dtm"], gt, valid, s, spec.alpha, gsd=g["gsd"], pred_ground=pg,
                                       before=before, p_edit=res.get("p_edit"))
        if before is not None:
            summary["lasground_new"] = dtm_metrics(before, gt, valid & np.isfinite(before), s, spec.alpha,
                                                   gsd=g["gsd"])
            true_dz = gt - before
    if "dz_before" in res:
        rows = block_priorities(res["dz_before"], g["gsd"], args.block_m, res.get("std"), true_dz, spec.alpha,
                                res.get("p_edit"), g["xmin"], g["ymax"])
        with open(out_dir / "priority.csv", "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        if true_dz is not None:
            summary["priority"] = capture_at(rows)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint")
    src.add_argument("--onnx")
    ap.add_argument("--scenes", type=Path, required=True, help="preprocessed scene root")
    ap.add_argument("--split-file", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stride", type=int, help="tile stride in pixels (default tile/2 = 50%% overlap)")
    ap.add_argument("--blend", choices=["min", "linear", "mean"], default="linear",
                    help="overlap blending (paper: min = best RMSE, linear = best balance)")
    ap.add_argument("--prior", choices=["auto", "global", "channel", "none"], default="auto")
    ap.add_argument("--init", choices=["dsm_noise", "noise", "dsm", "prior", "prior_noise", "dsm_q", "prior_q"],
                    help="override sampler init (see diffusion.py)")
    ap.add_argument("--t-start", type=int)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--tta", action="store_true", help="average the 8 flips/rotations")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-m", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-overlays", action="store_true", help="skip the coloured RGBA overlays")
    ap.add_argument("--max-scenes", type=int, help="evaluate only the first N scenes (quicker check)")
    ap.add_argument("--include-suspect", action="store_true", help="also evaluate scenes the quality gate flagged")
    a = ap.parse_args(argv)

    net, spec = load_net(a.checkpoint, a.onnx, a.device)
    scenes = sorted(p.parent for p in a.scenes.glob("*/meta.json"))
    if a.split_file:
        keep = set(json.loads(a.split_file.read_text())[a.split])
        scenes = [s for s in scenes if s.name in keep]
    if not a.include_suspect:
        sus = [s for s in scenes if json.loads((s / "meta.json").read_text()).get("quality", {}).get("suspect")]
        if sus:
            print(f"[info] skipping {len(sus)} scenes flagged suspect by preprocess (--include-suspect to keep)")
            scenes = [s for s in scenes if s not in sus]
    if a.max_scenes:
        scenes = scenes[:a.max_scenes]
    print(f"{len(scenes)} scenes, {a.samples} sample(s) each")
    all_rows = []
    for sd in scenes:
        summ = run_scene(sd, net, spec, a.out / sd.name, a)
        all_rows.append(summ)
        m = summ.get("model", {})
        b = summ.get("lasground_new", {})
        print(f"{sd.name}: model RMSE {m.get('rmse', float('nan')):.3f} m"
              + (f" | lasground_new {b.get('rmse', float('nan')):.3f} m" if b else ""))
    (a.out / "summary.json").write_text(json.dumps(all_rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

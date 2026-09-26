"""Predict DTMs for whole scenes, write GeoTIFFs, rank areas for editing.

    python -m groundiff.infer --checkpoint runs/before_after/best.pt \
        --scenes data/scenes_05m --split-file data/split.json --split test --out results/test
    python -m groundiff.infer --onnx models/before_after.onnx --scenes ... --out ...

Outputs per scene (GeoTIFF, same grid as the inputs):
    dtm.tif          predicted DTM (m)
    p_ground.tif     GrounDiff ground confidence sigmoid(l) (0-1)
    std.tif          spread across samples / TTA views (m), with --samples > 1 or --tta
    dz_before.tif    predicted DTM - lasground_new DTM (m): the predicted edit
    error.tif        predicted DTM - reference DTM (m), when a reference exists
and, when a reference DTM exists, metrics.json (model and lasground_new vs
reference; GrounDiff's RMSE/MAE/Type I/II/total plus ResDepth's MedAE/NMAD).

priority.csv ranks square blocks (default 100 m) by predicted edit size,
mean |dz_before| inside the block. With a reference it also reports the true
edit size and, in summary.json, how much of the true edit volume the top 5,
10 and 20 % of blocks capture. This ranking is our addition; neither paper
defines one.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .io_raster import write_geotiff
from .metrics import dtm_metrics
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
    model, cfg, _ = load_model(checkpoint, dev)
    return TorchNet(model, dev), RuntimeSpec.from_config(cfg)


def block_priorities(dz: np.ndarray, gsd: float, block_m: float, std: np.ndarray | None = None,
                     true_dz: np.ndarray | None = None, alpha: float = 0.2) -> list[dict]:
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
            row = {"row": r, "col": c, "pred_edit_mean_m": float(ad.mean()),
                   "pred_edit_frac": float((ad > alpha).mean()), "n": int(ok.sum())}
            if std is not None:
                row["std_mean_m"] = float(np.nanmean(std[r:r + b, c:c + b]))
            if true_dz is not None:
                td = np.abs(true_dz[r:r + b, c:c + b])
                row["true_edit_mean_m"] = float(np.nanmean(td)) if np.isfinite(td).any() else float("nan")
                row["true_edit_volume"] = float(np.nansum(td)) * gsd * gsd
            rows.append(row)
    rows.sort(key=lambda x: -x["pred_edit_mean_m"])
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


def run_scene(scene_dir: Path, net, spec: RuntimeSpec, out_dir: Path, args) -> dict:
    meta = json.loads((scene_dir / "meta.json").read_text())
    g = meta["grid"]
    names = set(spec.needed_channels) | {"gt_dtm", "gt_valid"}
    arrs = {n: np.load(scene_dir / f"{n}.npy") for n in names if (scene_dir / f"{n}.npy").exists()}
    res = predict_scene(arrs, spec, net, stride=args.stride, blend=args.blend, prior=args.prior,
                        init=args.init, t_start=args.t_start, n_samples=args.samples, tta=args.tta,
                        batch_size=args.batch_size, seed=args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    geo = (g["xmin"], g["ymax"], g["gsd"], meta.get("crs_wkt"))
    for k in ("dtm", "p_ground", "std", "dz_before", "prior"):
        if k in res:
            write_geotiff(out_dir / f"{k}.tif", res[k], *geo)
    summary = {"scene": meta["scene"]}
    true_dz = None
    if "gt_dtm" in arrs:
        gt = np.where(arrs["gt_valid"] > 0.5, arrs["gt_dtm"], np.nan)
        write_geotiff(out_dir / "error.tif", res["dtm"] - gt, *geo)
        valid = np.isfinite(gt) & np.isfinite(res["dtm"])
        s = arrs[spec.gate_channel]
        pg = res["p_ground"] > 0.5 if "p_ground" in res else None
        summary["model"] = dtm_metrics(res["dtm"], gt, valid, s, spec.alpha, gsd=g["gsd"], pred_ground=pg)
        if spec.prior_channel and spec.prior_channel in arrs:
            before = arrs[spec.prior_channel]
            summary["lasground_new"] = dtm_metrics(before, gt, valid & np.isfinite(before), s, spec.alpha,
                                                   gsd=g["gsd"])
            true_dz = gt - before
    if "dz_before" in res:
        rows = block_priorities(res["dz_before"], g["gsd"], args.block_m, res.get("std"), true_dz, spec.alpha)
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
    ap.add_argument("--blend", choices=["min", "linear", "mean"], default="min")
    ap.add_argument("--prior", choices=["auto", "global", "channel", "none"], default="auto")
    ap.add_argument("--init", help="override sampler init (see diffusion.py)")
    ap.add_argument("--t-start", type=int)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--tta", action="store_true", help="average the 8 flips/rotations")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-m", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    net, spec = load_net(a.checkpoint, a.onnx, a.device)
    scenes = sorted(p.parent for p in a.scenes.glob("*/meta.json"))
    if a.split_file:
        keep = set(json.loads(a.split_file.read_text())[a.split])
        scenes = [s for s in scenes if s.name in keep]
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

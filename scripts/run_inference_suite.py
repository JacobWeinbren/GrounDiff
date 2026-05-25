#!/usr/bin/env python3
"""Full inference suite — ONE PrioStitch pass per scene, all outputs.

For each scene this runs PrioStitch (+ optional D4 TTA) exactly once and
emits, from that single prediction:
  * <out>/scenes/<scene>_elevation.png   DSM / GT-DTM / pred-DTM panel
  * <out>/scenes/<scene>_analysis.png    residual / sigma(l) / error panel
  * <out>/scenes/<scene>_metrics.csv     per-scene metrics
  * <out>/scenes/<scene>_inference.npz   dtm_pred / logit / prob_ground
  * <out>/<scene>.laz                    classified LAZ (if --laz-root given
                                         and the source .laz is found)
  * <out>/rollup.csv                     per-scene rollup (all scenes)
  * <out>/summary.json + stdout          POOLED, denominator-gated headline

The pooled metrics use the same PooledAggregator as eval_priostitch_tta
(total-error / total-cells, robust to tiny-denominator scenes). No data is
filled, dropped, or fabricated — selection only chooses which scenes run.

Reuses verified code: priostitch_infer, visualize.make_all,
infer_to_laz.{bilinear_sample,_write_laz_with_classification}, and the
PooledAggregator + scene sampler from eval_priostitch_tta.

Usage (representative subset for the heavy image+LAZ outputs):
    python -m stage2_raster.scripts.run_inference_suite \
        --config   stage2_raster/configs/defra.json \
        --ckpt     /root/work/runs/run/best.pt \
        --val_root /root/work/runs/preprocessed_05m \
        --laz-root /root/work/laz \
        --n-scenes 60 --tta 8 --bf16 \
        --out      /root/work/runs/run/inference_suite

Set --n-scenes -1 to run every scene under val_root (heavy: LAZ + 2 PNGs
each). For the pooled headline number alone over 400 scenes without LAZ,
prefer eval_priostitch_tta.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage2_raster.utils.priostitch import priostitch_infer          # noqa: E402
from stage2_raster.scripts.visualize import make_all                  # noqa: E402
from stage2_raster.scripts.eval_priostitch_tta import (               # noqa: E402
    PooledAggregator, stratified_scene_sample, build_and_load, _nullctx)
from stage2_raster.scripts.national_error_map import (                # noqa: E402
    make_record, render_maps, write_raw)


def _find_laz(laz_root: Path | None, scene: str, pattern: str) -> Path | None:
    """Resolve a scene's source LAZ. `pattern` may contain {scene}.
    Falls back to a recursive glob for <scene>*.la[sz] if the direct
    path isn't there. Returns None if nothing matches."""
    if laz_root is None:
        return None
    direct = laz_root / pattern.format(scene=scene)
    if direct.exists():
        return direct
    for ext in ('laz', 'las'):
        hits = sorted(laz_root.rglob(f"{scene}*.{ext}"))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--ckpt', required=True, type=Path)
    ap.add_argument('--val_root', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--n-scenes', type=int, default=60,
                    help='-1 = every scene under val_root')
    ap.add_argument('--n-bins', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tta', type=int, default=8, choices=[1, 4, 8])
    ap.add_argument('--tile-size', type=int, default=256)
    ap.add_argument('--overlap', type=int, default=128)
    ap.add_argument('--coarse-size', type=int, default=256)
    ap.add_argument('--blend-mode', type=str, default='linear')
    ap.add_argument('--alpha-metres', type=float, default=0.20)
    ap.add_argument('--min-classify-n', type=int, default=1000)
    ap.add_argument('--laz-root', type=Path, default=None,
                    help='dir holding source LAZ; omit to skip LAZ output')
    ap.add_argument('--laz-pattern', type=str, default='{scene}.laz')
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--no-images', action='store_true')
    ap.add_argument('--no-npz', action='store_true')
    ap.add_argument('--map-every', type=int, default=0,
                    help='redraw the national England error maps every N '
                         'scored scenes (live preview). 0 = only at the end '
                         'if --map-out set. Renders the 5 default metric maps '
                         'from results so far.')
    ap.add_argument('--map-out', type=Path, default=None,
                    help='dir for the live national maps + raw data. Defaults '
                         'to <out>/national_map when --map-every > 0.')
    ap.add_argument('--map-boundary', type=Path, default=None,
                    help='optional EPSG:27700 GeoJSON coastline overlay')
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--no-ema', action='store_true')
    args = ap.parse_args()

    out = args.out
    (out / 'scenes').mkdir(parents=True, exist_ok=True)
    _logf = open(out / 'inference.log', 'a')

    def log(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg, flush=True)
        _logf.write(msg + '\n'); _logf.flush()

    log(f"# === full inference suite ===  tta={args.tta}  "
        f"tile={args.tile_size} overlap={args.overlap} "
        f"coarse={args.coarse_size} blend={args.blend_mode}  "
        f"laz={'on' if args.laz_root else 'off'}")

    model, cfg, step = build_and_load(
        args.config, args.ckpt, args.device,
        use_ema=not args.no_ema, log=log)

    if args.n_scenes is not None and args.n_scenes < 0:
        scenes = sorted(p.parent.name
                        for p in args.val_root.glob('*/raster.npz'))
        log(f"# running ALL {len(scenes)} scenes")
    else:
        scenes = stratified_scene_sample(
            args.val_root, args.n_scenes, n_bins=args.n_bins,
            seed=args.seed, log=log)

    # LAZ helpers (imported lazily so a missing laspy doesn't block
    # the image/CSV path).
    laz_fns = None
    if args.laz_root is not None:
        try:
            from stage2_raster.scripts.infer_to_laz import (
                bilinear_sample, _write_laz_with_classification)
            import laspy  # noqa: F401
            laz_fns = (bilinear_sample, _write_laz_with_classification)
        except Exception as e:  # noqa: BLE001
            log(f"# WARN: LAZ output disabled (import failed: {e})")

    amp_dtype = torch.bfloat16 if args.bf16 else None
    agg = PooledAggregator()
    rollup_csv = out / 'rollup.csv'
    if rollup_csv.exists():
        rollup_csv.unlink()

    # Live national-map state.
    map_records = []
    map_out = (args.map_out if args.map_out is not None
               else (out / 'national_map' if args.map_every > 0 else None))
    if map_out is not None:
        map_out.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm.auto import tqdm
        it = tqdm(scenes, desc=f"infer+tta{args.tta}", unit="scene")
    except ImportError:
        it = scenes

    t0 = time.time(); n_ok = 0; n_laz = 0
    for name in it:
        rp = args.val_root / name / 'raster.npz'
        if not rp.exists():
            log(f"# missing {rp}, skip"); continue
        try:
            with np.load(str(rp)) as z:
                dsm_max = z['dsm_max']; dsm_min = z['dsm_min']
                dsm_mean = z['dsm_mean']; dsm_mask = z['dsm_mask']
                gt_dtm = z['gt_dtm']; valid = z['valid']
                had_g = z['had_ground_return'] if \
                    'had_ground_return' in z.files else None
                gsd = float(z['gsd']) if 'gsd' in z.files else 0.5
                alpha = float(z['alpha_metres']) if \
                    'alpha_metres' in z.files else args.alpha_metres
                bbox = tuple(z['bbox'].tolist()) if 'bbox' in z.files else None

            cm = (torch.autocast(device_type=str(args.device).split(':')[0],
                                 dtype=amp_dtype)
                  if amp_dtype is not None else _nullctx())
            with torch.inference_mode(), cm:
                res = priostitch_infer(
                    model, dsm_max, dsm_min, dsm_mean,
                    dsm_mask.astype(np.float32),
                    coarse_size=args.coarse_size, tile_size=args.tile_size,
                    overlap=args.overlap, blend_mode=args.blend_mode,
                    device=args.device, tta=args.tta)
            dtm_pred = res['dtm_pred']; logit = res['logit']
            prob_g = res['prob_ground']

            prefix = out / 'scenes' / name
            if not args.no_npz:
                np.savez_compressed(
                    str(prefix.with_name(name + '_inference.npz')),
                    dtm_pred=dtm_pred.astype(np.float32),
                    logit=logit.astype(np.float32),
                    prob_ground=prob_g.astype(np.float32))

            if not args.no_images:
                m = make_all(
                    dsm_max=dsm_max, gt_dtm=gt_dtm, valid_gt=valid,
                    dsm_mask=dsm_mask, dtm_pred=dtm_pred,
                    had_ground=had_g, prob_ground=prob_g, logit=logit,
                    gsd=gsd, alpha_metres=alpha,
                    out_prefix=prefix, dpi=args.dpi, scene_name=name,
                    rollup_csv=rollup_csv)
            else:
                from stage2_raster.scripts.visualize import (
                    compute_metrics, append_metrics_csv)
                m = compute_metrics(
                    dsm_max=dsm_max, gt_dtm=gt_dtm, valid_gt=valid,
                    dsm_mask=dsm_mask, had_ground=had_g,
                    dtm_pred=dtm_pred, prob_ground=prob_g, alpha_metres=alpha)
                append_metrics_csv(rollup_csv, m, scene=name)
            agg.add(m)

            # Accumulate a national-map record (footprint + metrics) and,
            # every --map-every scenes, redraw the live England maps from
            # everything scored so far so progress is previewable.
            if map_out is not None and bbox is not None:
                map_records.append(make_record(name, bbox, m))
                if (args.map_every > 0
                        and len(map_records) % args.map_every == 0):
                    try:
                        render_maps(map_records, map_out,
                                    boundary=args.map_boundary, quiet=True,
                                    title_note=f'live: {len(map_records)} tiles')
                        write_raw(map_records, map_out)
                        log(f"# [map] refreshed national maps "
                            f"({len(map_records)} tiles) -> {map_out}")
                    except Exception as e:  # noqa: BLE001
                        log(f"# [map] refresh failed: {type(e).__name__}: {e}")

            # ---- LAZ (single-pass: reuse dtm_pred we already computed) ----
            if laz_fns is not None and bbox is not None:
                laz_path = _find_laz(args.laz_root, name, args.laz_pattern)
                if laz_path is None:
                    log(f"# [laz] no source LAZ for {name}, skip")
                else:
                    try:
                        import laspy
                        bilinear_sample, write_laz = laz_fns
                        with laspy.open(str(laz_path)) as f:
                            las = f.read()
                        xyz = np.column_stack(
                            [las.x, las.y, las.z]).astype(np.float64)
                        z_at = bilinear_sample(dtm_pred, xyz[:, 0],
                                               xyz[:, 1], bbox, gsd)
                        resid = xyz[:, 2].astype(np.float32) - z_at
                        is_g = np.abs(resid) < alpha
                        cls_new = np.where(is_g, 2, 1).astype(np.uint8)
                        p_at = bilinear_sample(prob_g, xyz[:, 0],
                                               xyz[:, 1], bbox, gsd)
                        p_u16 = np.clip(p_at * 65535.0, 0,
                                        65535).astype(np.uint16)
                        write_laz(
                            laz_path, out / f"{name}.laz",
                            cls_new=cls_new, dtm_pred_z=z_at.astype(np.float32),
                            residual_z=resid.astype(np.float32),
                            prob_ground_u16=p_u16)
                        n_laz += 1
                    except Exception as e:  # noqa: BLE001
                        log(f"# [laz] FAIL {name}: {type(e).__name__}: {e}")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            log(f"# FAIL {name}: {type(e).__name__}: {e}")
            continue

    dt = time.time() - t0
    log(f"# scored {n_ok}/{len(scenes)} scenes, wrote {n_laz} LAZ, "
        f"in {dt/60:.1f} min ({dt/max(n_ok,1):.1f}s/scene)")

    # Final national map + raw data from all scored scenes.
    if map_out is not None and map_records:
        try:
            paths = render_maps(map_records, map_out,
                                boundary=args.map_boundary,
                                title_note=f'final: {len(map_records)} tiles')
            write_raw(map_records, map_out)
            log(f"# [map] final national maps ({len(map_records)} tiles) -> "
                f"{map_out}  ({len(paths)} PNGs + raw data)")
        except Exception as e:  # noqa: BLE001
            log(f"# [map] final render failed: {type(e).__name__}: {e}")

    pooled = agg.result()
    summary = dict(ckpt=str(args.ckpt), step=step, tta=args.tta,
                   n_scenes_scored=n_ok, n_laz=n_laz, pooled=pooled)
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    log("")
    log("================  POOLED (PrioStitch"
        + (f"+TTA{args.tta}" if args.tta > 1 else "") + ")  ================")
    log(f"  scenes scored : {n_ok}   LAZ written: {n_laz}")
    log(f"  E_T1 (residual): {pooled['e_t1_pct']:6.2f} %")
    log(f"  E_T2 (residual): {pooled['e_t2_pct']:6.2f} %")
    log(f"  E_tot(residual): {pooled['e_tot_pct']:6.2f} %")
    log(f"  RMSE          : {pooled['rmse_m']:.3f} m")
    log(f"  within 20cm   : {100*pooled['frac_within_20cm']:.1f} %")
    log(f"  outputs       : {out}/scenes/*.png|*.csv|*.npz, {out}/*.laz, "
        f"{out}/rollup.csv, {out}/summary.json")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""PrioStitch + TTA evaluation over a stratified scene set.

This is the *headline* eval: it runs the full PrioStitch inference path
(coarse global prior -> overlapping fine tiles -> blend), optionally with
D4 test-time augmentation, on a coverage-stratified sample of scenes, and
reports POOLED, denominator-gated metrics.

Why this exists (see the debugging history):
  * The per-tile `evaluate()` in train.py runs single tiles with no global
    prior. That is the paper's *weakest* configuration (supp. §12) and is
    dominated by sparse, context-starved tiles -> inflated E_tot (~14%).
  * The paper's headline numbers are all PrioStitch. This harness measures
    THAT, so the reported number reflects deployment.
  * It pools FP/FN/SSE across scenes (total errors / total cells) instead
    of averaging per-scene percentages. Averaging per-scene E_T1 is
    meaningless when a near-all-ground scene has ~10^3 non-ground cells:
    a few hundred FP becomes 70%+ on a tiny base (Bernoulli noise). The
    `--min-classify-n` gate suppresses the per-scene RATIO in that regime
    (reported as NaN) but every pixel still counts in the pooled E_tot and
    RMSE -- no data is dropped or invented.

Data-integrity note: this script does not fill, drop, or fabricate any
data. No-data handling is whatever the model/inference path already does
(paper-faithful invalid->0). Stratification only chooses WHICH scenes are
scored; it never alters a scene.

Usage:
    python -m stage2_raster.scripts.eval_priostitch_tta \
        --config   stage2_raster/configs/defra.json \
        --ckpt     /root/work/runs/run/best.pt \
        --val_root /root/work/runs/preprocessed_05m \
        --n-scenes 400 --tta 8 \
        --out      /root/work/runs/run/eval_priostitch_tta

Outept:
    <out>/rollup.csv          per-scene metrics (TTA, PrioStitch)
    <out>/summary.json        pooled headline metrics
    stdout                    pooled summary table
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------- #
#  Scene discovery + coverage-stratified sampling
# --------------------------------------------------------------------- #
def _scene_coverage(raster_path: Path) -> float | None:
    """Cheap coverage probe: fraction of valid-GT cells that have a DSM
    return. Reads only the two mask arrays from the npz (lazy), not the
    full elevation rasters. Returns None if the scene can't be read."""
    try:
        with np.load(str(raster_path)) as z:
            if 'dsm_mask' not in z.files or 'valid' not in z.files:
                return None
            dmask = z['dsm_mask'].astype(bool)
            vmask = z['valid'].astype(bool)
        nv = int(vmask.sum())
        if nv == 0:
            return 0.0
        return float((dmask & vmask).sum()) / float(nv)
    except Exception:
        return None


def stratified_scene_sample(val_root: Path, n_scenes: int, *,
                             n_bins: int = 5, seed: int = 0,
                             log=print) -> list[str]:
    """Pick `n_scenes` scene names stratified by coverage.

    We bin scenes into `n_bins` equal-width coverage bands over [0, 1] and
    sample from each band in proportion to its population, but with a floor
    so the sparse tail (where the model is weakest) is always represented.
    If fewer scenes exist than requested, all are returned.

    Stratifying by coverage matters because coverage is the single
    strongest predictor of error in this data (corr(coverage, RMSE) ~ -0.45
    in the diagnostics). A 'representative' set must span the coverage
    distribution rather than over-sampling easy high-coverage scenes.
    """
    scenes = sorted(p.parent.name for p in val_root.glob('*/raster.npz'))
    if not scenes:
        raise SystemExit(f"no scenes found under {val_root}/*/raster.npz")
    if len(scenes) <= n_scenes:
        log(f"# {len(scenes)} scenes available <= {n_scenes} requested; "
            f"using all")
        return scenes

    log(f"# probing coverage for {len(scenes)} scenes for stratification...")
    covs = {}
    for i, name in enumerate(scenes):
        c = _scene_coverage(val_root / name / 'raster.npz')
        if c is not None:
            covs[name] = c
        if (i + 1) % 500 == 0:
            log(f"#   probed {i + 1}/{len(scenes)}")
    named = list(covs.items())

    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[list[str]] = [[] for _ in range(n_bins)]
    for name, c in named:
        b = min(int(np.searchsorted(edges, c, side='right') - 1), n_bins - 1)
        bins[max(b, 0)].append(name)

    # Proportional allocation with a per-bin floor so tails are covered.
    pops = np.array([len(b) for b in bins], dtype=float)
    floor = max(1, n_scenes // (n_bins * 4))  # ~5% of budget per non-empty bin
    alloc = np.zeros(n_bins, dtype=int)
    nonempty = pops > 0
    alloc[nonempty] = np.maximum(
        floor, np.round(n_scenes * pops[nonempty] / pops.sum())).astype(int)
    # Trim/pad to exactly n_scenes (can't exceed a bin's population).
    alloc = np.minimum(alloc, pops.astype(int))
    while alloc.sum() > n_scenes:
        i = int(np.argmax(alloc))
        alloc[i] -= 1
    while alloc.sum() < n_scenes:
        head = pops.astype(int) - alloc
        if head.max() <= 0:
            break
        i = int(np.argmax(head))
        alloc[i] += 1

    chosen: list[str] = []
    for b_idx, k in enumerate(alloc):
        if k <= 0 or not bins[b_idx]:
            continue
        pick = rng.choice(len(bins[b_idx]), size=int(k), replace=False)
        chosen.extend(bins[b_idx][j] for j in pick)
    log(f"# stratified sample: {len(chosen)} scenes across {n_bins} "
        f"coverage bins (alloc={alloc.tolist()}, "
        f"bin edges={[round(e, 2) for e in edges]})")
    return sorted(chosen)


# --------------------------------------------------------------------- #
#  Pooled aggregator (counts only -> robust pooled rates)
# --------------------------------------------------------------------- #
class PooledAggregator:
    """Accumulate raw counts/SSE across scenes; report pooled rates.

    Pooled E_T1/E_T2/E_tot are total-error / total-cells (the paper's
    convention), so a few tiny-denominator scenes can't distort them.
    RMSE/within-Xcm are pixel-weighted. Per-class ratios are always
    reportable when pooled because the pooled denominators are huge.
    """

    def __init__(self):
        self.fp = 0; self.fn = 0          # residual method (primary)
        self.fp_l = 0; self.fn_l = 0      # sigma(l) method
        self.n_g = 0; self.n_ng = 0; self.n_v = 0
        self.sse = 0.0; self.sae = 0.0
        self.n_err = 0
        self.within20 = 0; self.within50 = 0

    def add(self, m: dict):
        self.fp += m['n_fp']; self.fn += m['n_fn']
        self.n_g += m['n_gt_ground']; self.n_ng += m['n_gt_non_ground']
        self.n_v += m['n_cells_cls_valid']
        ne = m['n_cells_err_valid']
        if ne > 0 and not math.isnan(m['rmse_m']):
            self.sse += (m['rmse_m'] ** 2) * ne
            self.sae += m['mae_m'] * ne
            self.within20 += m['frac_within_20cm'] * ne
            self.within50 += m['frac_within_50cm'] * ne
            self.n_err += ne
        if 'e_t1_pct_logit' in m:
            # back out logit FP/FN from the per-scene rates + denominators
            self.fp_l += round(m['e_t1_pct_logit'] / 100.0 * m['n_gt_non_ground'])
            self.fn_l += round(m['e_t2_pct_logit'] / 100.0 * m['n_gt_ground'])

    def result(self) -> dict:
        d = max(self.n_v, 1)
        out = dict(
            n_scenes_pixels=self.n_v,
            e_t1_pct=100.0 * self.fp / max(self.n_ng, 1),
            e_t2_pct=100.0 * self.fn / max(self.n_g, 1),
            e_tot_pct=100.0 * (self.fp + self.fn) / d,
            e_t1_pct_logit=100.0 * self.fp_l / max(self.n_ng, 1),
            e_t2_pct_logit=100.0 * self.fn_l / max(self.n_g, 1),
            e_tot_pct_logit=100.0 * (self.fp_l + self.fn_l) / d,
            rmse_m=math.sqrt(self.sse / max(self.n_err, 1)),
            mae_m=self.sae / max(self.n_err, 1),
            frac_within_20cm=self.within20 / max(self.n_err, 1),
            frac_within_50cm=self.within50 / max(self.n_err, 1),
        )
        return out


# --------------------------------------------------------------------- #
#  Model loading (mirrors train.py)
# --------------------------------------------------------------------- #
def build_and_load(config_path: Path, ckpt_path: Path, device: str,
                   use_ema: bool = True, log=print):
    import sys
    # Ensure the package root is importable when run as a file.
    pkg_root = Path(__file__).resolve().parents[2]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from stage2_raster.models import GrounDiffRaster

    with open(config_path) as f:
        cfg = json.load(f)

    def _strip_doc(d):
        return {k: v for k, v in d.items() if not k.startswith('_')} \
            if isinstance(d, dict) else d

    model = GrounDiffRaster(
        backbone=str(cfg.get('backbone', 'unet')),
        backbone_kwargs=_strip_doc(cfg.get('model', {})),
        diffusion_kwargs=_strip_doc(cfg.get('diffusion', {})),
        loss_kwargs=_strip_doc(cfg.get('loss', {})),
    ).to(device)

    def _clean(sd: dict) -> dict:
        """Strip wrapper prefixes so a checkpoint saved under torch.compile
        ('_orig_mod.'), DDP ('module.'), or nested in the parent model
        ('net.') loads into the bare `model.net`. The prefixes can STACK —
        e.g. the EMA shadow was built from the whole compiled model, so its
        keys look like 'net._orig_mod.input_blocks...'. We therefore strip
        repeatedly, in any order, until the key stops changing. Keys that
        don't belong to the net (e.g. diffusion-schedule buffers like
        'diffusion.betas') are left as-is and simply won't match net keys
        (harmless — they're filtered by the shape/name check at the call
        site)."""
        prefixes = ('_orig_mod.', 'module.', 'net.')
        out = {}
        for k, v in sd.items():
            kk = k
            changed = True
            while changed:
                changed = False
                for pre in prefixes:
                    if kk.startswith(pre):
                        kk = kk[len(pre):]
                        changed = True
            out[kk] = v
        return out

    d = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    sd = d.get('net', d.get('dit'))
    if sd is None:
        raise KeyError(f"{ckpt_path}: no 'net'/'dit' state-dict")
    sd = _clean(sd)
    missing, unexpected = model.net.load_state_dict(sd, strict=False)
    # Tolerate nothing real missing: only allow empty (the clean should be
    # exact). Surface anything genuinely unmatched.
    real_missing = [k for k in missing]
    real_unexpected = [k for k in unexpected]
    if real_missing or real_unexpected:
        log(f"# WARN load: {len(real_missing)} missing, "
            f"{len(real_unexpected)} unexpected after prefix-strip "
            f"(e.g. missing={real_missing[:2]}, "
            f"unexpected={real_unexpected[:2]})")
    step = int(d.get('step', -1))
    log(f"# loaded {ckpt_path.name} (step {step}), "
        f"{model.num_params / 1e6:.1f}M params")

    # Apply EMA weights if present (best-quality inference, matches train).
    if use_ema and 'ema' in d:
        try:
            shadow = d['ema'].get('shadow', d['ema'])
            shadow = _clean(shadow)
            msd = model.net.state_dict()
            applied = 0
            for k, v in shadow.items():
                if k in msd and msd[k].shape == v.shape:
                    msd[k] = v.to(msd[k].dtype); applied += 1
            model.net.load_state_dict(msd)
            log(f"# applied EMA weights ({applied}/{len(msd)} tensors)")
        except Exception as e:  # noqa: BLE001
            log(f"# WARN: could not apply EMA ({e}); using raw weights")
    model.eval()
    return model, cfg, step


# --------------------------------------------------------------------- #
#  Score one already-loaded model over a fixed scene list
# --------------------------------------------------------------------- #
def score_model(model, val_root: Path, scenes: list[str], *,
                priostitch_infer, compute_metrics,
                coarse_size: int, tile_size: int, overlap: int,
                blend_mode: str, device: str, tta: int,
                amp_dtype, alpha_default: float,
                min_classify_n: int, log=print,
                desc: str = "priostitch") -> tuple[dict, list[dict]]:
    """Run PrioStitch(+TTA) over `scenes`, return (pooled, rows).

    Shared by the single-checkpoint eval and the ranking mode so both
    use identical inference + metric logic on the SAME scene list (which
    matters for ranking: every checkpoint must see the same scenes)."""
    agg = PooledAggregator()
    rows: list[dict] = []
    n_ok = 0
    try:
        from tqdm.auto import tqdm
        it = tqdm(scenes, desc=desc, unit="scene", leave=False)
    except ImportError:
        it = scenes
    for name in it:
        rp = val_root / name / 'raster.npz'
        if not rp.exists():
            log(f"# missing {rp}, skipping"); continue
        try:
            with np.load(str(rp)) as z:
                dsm_max = z['dsm_max']; dsm_min = z['dsm_min']
                dsm_mean = z['dsm_mean']; dsm_mask = z['dsm_mask']
                gt_dtm = z['gt_dtm']; valid = z['valid']
                had_g = z['had_ground_return'] if \
                    'had_ground_return' in z.files else None
                alpha = float(z['alpha_metres']) if \
                    'alpha_metres' in z.files else alpha_default
            cm = (torch.autocast(device_type=str(device).split(':')[0],
                                 dtype=amp_dtype)
                  if amp_dtype is not None else _nullctx())
            with torch.inference_mode(), cm:
                res = priostitch_infer(
                    model, dsm_max, dsm_min, dsm_mean,
                    dsm_mask.astype(np.float32),
                    coarse_size=coarse_size, tile_size=tile_size,
                    overlap=overlap, blend_mode=blend_mode,
                    device=device, tta=tta)
            m = compute_metrics(
                dsm_max=dsm_max, gt_dtm=gt_dtm, valid_gt=valid,
                dsm_mask=dsm_mask, had_ground=had_g,
                dtm_pred=res['dtm_pred'], prob_ground=res['prob_ground'],
                alpha_metres=alpha)
            agg.add(m)
            row = dict(scene=name); row.update(m)
            if m['n_gt_non_ground'] < min_classify_n:
                row['e_t1_pct'] = float('nan')
                row['e_t1_pct_logit'] = float('nan')
            if m['n_gt_ground'] < min_classify_n:
                row['e_t2_pct'] = float('nan')
                row['e_t2_pct_logit'] = float('nan')
            rows.append(row); n_ok += 1
        except Exception as e:  # noqa: BLE001
            log(f"# FAIL {name}: {type(e).__name__}: {e}")
            continue
    pooled = agg.result(); pooled['n_scenes_scored'] = n_ok
    return pooled, rows


def rank_checkpoints(ckpts: list[Path], args, *, log) -> Path:
    """Evaluate each checkpoint under PrioStitch+TTA on the SAME stratified
    scene set, rank by pooled e_tot (or args.rank_metric), write a ranking
    CSV/JSON, and return the winning checkpoint path."""
    import sys
    pkg_root = Path(__file__).resolve().parents[2]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from stage2_raster.utils.priostitch import priostitch_infer
    from stage2_raster.scripts.visualize import compute_metrics

    # One scene set, fixed across all checkpoints (same seed/strata).
    scenes = stratified_scene_sample(
        args.val_root, args.n_scenes, n_bins=args.n_bins,
        seed=args.seed, log=log)
    amp_dtype = torch.bfloat16 if args.bf16 else None
    metric = args.rank_metric
    lower_is_better = metric not in ('frac_within_20cm', 'frac_within_50cm')

    results = []
    for cp in ckpts:
        log(f"\n# ---- ranking checkpoint: {cp.name} ----")
        model, _cfg, step = build_and_load(
            args.config, cp, args.device,
            use_ema=not args.no_ema, log=log)
        pooled, _rows = score_model(
            model, args.val_root, scenes,
            priostitch_infer=priostitch_infer, compute_metrics=compute_metrics,
            coarse_size=args.coarse_size, tile_size=args.tile_size,
            overlap=args.overlap, blend_mode=args.blend_mode,
            device=args.device, tta=args.tta, amp_dtype=amp_dtype,
            alpha_default=args.alpha_metres, min_classify_n=args.min_classify_n,
            log=log, desc=cp.stem)
        pooled['ckpt'] = str(cp); pooled['step'] = step
        results.append(pooled)
        log(f"#   {cp.name}: pooled e_tot={pooled['e_tot_pct']:.3f}%  "
            f"rmse={pooled['rmse_m']:.3f}m  "
            f"e_t2={pooled['e_t2_pct']:.3f}%  "
            f"within20={100*pooled['frac_within_20cm']:.1f}%")
        del model
        if str(args.device).startswith('cuda'):
            torch.cuda.empty_cache()

    results.sort(key=lambda r: r[metric], reverse=not lower_is_better)
    best = results[0]

    args.out.mkdir(parents=True, exist_ok=True)
    import csv
    with open(args.out / 'ranking.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'ckpt', 'step', metric, 'e_tot_pct', 'e_t1_pct',
                    'e_t2_pct', 'rmse_m', 'mae_m', 'frac_within_20cm',
                    'n_scenes_scored'])
        for i, r in enumerate(results):
            w.writerow([i + 1, r['ckpt'], r['step'], f"{r[metric]:.6f}",
                        f"{r['e_tot_pct']:.4f}", f"{r['e_t1_pct']:.4f}",
                        f"{r['e_t2_pct']:.4f}", f"{r['rmse_m']:.6f}",
                        f"{r['mae_m']:.6f}", f"{r['frac_within_20cm']:.6f}",
                        r['n_scenes_scored']])
    with open(args.out / 'ranking.json', 'w') as f:
        json.dump(dict(rank_metric=metric, results=results,
                       best=best['ckpt']), f, indent=2)

    log(f"\n================  CHECKPOINT RANKING "
        f"(by pooled {metric}, lower={'better' if lower_is_better else 'worse'})"
        f"  ================")
    for i, r in enumerate(results):
        mark = '  <-- BEST' if i == 0 else ''
        log(f"  {i+1}. {Path(r['ckpt']).name:28s} "
            f"{metric}={r[metric]:.4f}  e_tot={r['e_tot_pct']:.3f}%  "
            f"rmse={r['rmse_m']:.3f}m{mark}")
    log(f"\n# BEST -> {best['ckpt']}")
    log(f"# wrote {args.out/'ranking.csv'} and ranking.json")
    # Also drop the winner's path as a plain file for easy shell capture.
    (args.out / 'best_ckpt.txt').write_text(best['ckpt'] + '\n')
    return Path(best['ckpt'])


# --------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--ckpt', type=Path, default=None,
                    help='single checkpoint to evaluate (omit when using '
                         '--rank-glob)')
    ap.add_argument('--rank-glob', type=str, default=None,
                    help='glob of checkpoints to rank under PrioStitch+TTA '
                         "(e.g. '/root/work/runs/run/ckpt_step*.pt'). "
                         'Evaluates each on the SAME scene set, picks best.')
    ap.add_argument('--rank-last', type=int, default=5,
                    help='with --rank-glob, only rank the last N by step')
    ap.add_argument('--rank-metric', type=str, default='e_tot_pct',
                    choices=['e_tot_pct', 'rmse_m', 'e_t2_pct', 'mae_m',
                             'frac_within_20cm'],
                    help='pooled metric to rank by (default e_tot_pct)')
    ap.add_argument('--val_root', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--n-scenes', type=int, default=400)
    ap.add_argument('--n-bins', type=int, default=5,
                    help='coverage strata for representative sampling')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tta', type=int, default=8, choices=[1, 4, 8],
                    help='D4 test-time augmentation passes per tile')
    ap.add_argument('--tile-size', type=int, default=256)
    ap.add_argument('--overlap', type=int, default=128)
    ap.add_argument('--coarse-size', type=int, default=256)
    ap.add_argument('--blend-mode', type=str, default='linear')
    ap.add_argument('--min-classify-n', type=int, default=1000,
                    help='per-scene E_T1/E_T2 ratio is NaN below this '
                         'denominator (pixels still count in pooled)')
    ap.add_argument('--alpha-metres', type=float, default=0.20)
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--bf16', action='store_true',
                    help='run inference under bf16 autocast')
    ap.add_argument('--no-ema', action='store_true')
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / 'eval.log'
    _logf = open(log_path, 'a')

    def log(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg, flush=True)
        _logf.write(msg + '\n'); _logf.flush()

    import sys
    pkg_root = Path(__file__).resolve().parents[2]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from stage2_raster.utils.priostitch import priostitch_infer
    from stage2_raster.scripts.visualize import compute_metrics

    # ---- ranking mode: evaluate several checkpoints, pick the best ----
    if args.rank_glob:
        import glob as _glob, re as _re
        cks = [Path(p) for p in _glob.glob(args.rank_glob)]
        if not cks:
            raise SystemExit(f"--rank-glob matched nothing: {args.rank_glob}")
        def _stepnum(p):
            m = _re.search(r'(\d+)', p.stem)
            return int(m.group(1)) if m else -1
        cks = sorted(cks, key=_stepnum)
        if args.rank_last and args.rank_last > 0:
            cks = cks[-args.rank_last:]
        log(f"# ranking {len(cks)} checkpoints by pooled {args.rank_metric}: "
            f"{[c.name for c in cks]}")
        rank_checkpoints(cks, args, log=log)
        return

    if args.ckpt is None:
        raise SystemExit("provide --ckpt (single eval) or --rank-glob (ranking)")

    log(f"# === PrioStitch+TTA eval ===  tta={args.tta}  "
        f"tile={args.tile_size} overlap={args.overlap} "
        f"coarse={args.coarse_size} blend={args.blend_mode}")

    model, cfg, step = build_and_load(
        args.config, args.ckpt, args.device,
        use_ema=not args.no_ema, log=log)

    scenes = stratified_scene_sample(
        args.val_root, args.n_scenes, n_bins=args.n_bins,
        seed=args.seed, log=log)

    amp_dtype = torch.bfloat16 if args.bf16 else None
    agg = PooledAggregator()
    rows: list[dict] = []
    t0 = time.time()
    n_ok = 0

    try:
        from tqdm.auto import tqdm
        it = tqdm(scenes, desc=f"priostitch+tta{args.tta}", unit="scene")
    except ImportError:
        it = scenes

    for name in it:
        rp = args.val_root / name / 'raster.npz'
        if not rp.exists():
            log(f"# missing {rp}, skipping"); continue
        try:
            with np.load(str(rp)) as z:
                dsm_max = z['dsm_max']; dsm_min = z['dsm_min']
                dsm_mean = z['dsm_mean']; dsm_mask = z['dsm_mask']
                gt_dtm = z['gt_dtm']; valid = z['valid']
                had_g = z['had_ground_return'] if \
                    'had_ground_return' in z.files else None
                alpha = float(z['alpha_metres']) if \
                    'alpha_metres' in z.files else args.alpha_metres

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

            m = compute_metrics(
                dsm_max=dsm_max, gt_dtm=gt_dtm, valid_gt=valid,
                dsm_mask=dsm_mask, had_ground=had_g,
                dtm_pred=res['dtm_pred'], prob_ground=res['prob_ground'],
                alpha_metres=alpha)
            agg.add(m)

            # Per-scene row: gate the unstable per-class ratios.
            row = dict(scene=name)
            row.update(m)
            if m['n_gt_non_ground'] < args.min_classify_n:
                row['e_t1_pct'] = float('nan')
                row['e_t1_pct_logit'] = float('nan')
            if m['n_gt_ground'] < args.min_classify_n:
                row['e_t2_pct'] = float('nan')
                row['e_t2_pct_logit'] = float('nan')
            rows.append(row)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            log(f"# FAIL {name}: {type(e).__name__}: {e}")
            continue

    dt = time.time() - t0
    log(f"# scored {n_ok}/{len(scenes)} scenes in {dt/60:.1f} min "
        f"({dt/max(n_ok,1):.1f}s/scene)")

    # ---- write per-scene rollup ----
    import csv
    rollup = out / 'rollup.csv'
    if rows:
        keys = list(rows[0].keys())
        with open(rollup, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        log(f"# wrote {rollup}")

    # ---- pooled headline ----
    pooled = agg.result()
    # robust per-scene medians (outlier-resistant companions to pooled)
    def _med(key):
        vals = [r[key] for r in rows
                if key in r and not (isinstance(r[key], float)
                                     and math.isnan(r[key]))]
        return float(np.median(vals)) if vals else float('nan')

    summary = dict(
        ckpt=str(args.ckpt), step=step, tta=args.tta,
        n_scenes_scored=n_ok,
        pooled=pooled,
        median_per_scene=dict(
            rmse_m=_med('rmse_m'),
            e_tot_pct=_med('e_tot_pct'),
            frac_within_20cm=_med('frac_within_20cm'),
        ),
        config=dict(tile_size=args.tile_size, overlap=args.overlap,
                    coarse_size=args.coarse_size, blend_mode=args.blend_mode,
                    min_classify_n=args.min_classify_n),
    )
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    log("")
    log("================  POOLED (PrioStitch"
        + (f"+TTA{args.tta}" if args.tta > 1 else "") + ")  ================")
    log(f"  scenes scored : {n_ok}")
    log(f"  E_T1  (residual): {pooled['e_t1_pct']:6.2f} %   "
        f"[sigma(l): {pooled['e_t1_pct_logit']:6.2f} %]")
    log(f"  E_T2  (residual): {pooled['e_t2_pct']:6.2f} %   "
        f"[sigma(l): {pooled['e_t2_pct_logit']:6.2f} %]")
    log(f"  E_tot (residual): {pooled['e_tot_pct']:6.2f} %   "
        f"[sigma(l): {pooled['e_tot_pct_logit']:6.2f} %]")
    log(f"  RMSE          : {pooled['rmse_m']:.3f} m")
    log(f"  MAE           : {pooled['mae_m']:.3f} m")
    log(f"  within 20cm   : {100*pooled['frac_within_20cm']:.1f} %")
    log(f"  within 50cm   : {100*pooled['frac_within_50cm']:.1f} %")
    log(f"  --- median per-scene (outlier-robust) ---")
    log(f"  RMSE  median  : {summary['median_per_scene']['rmse_m']:.3f} m")
    log(f"  E_tot median  : {summary['median_per_scene']['e_tot_pct']:.2f} %")
    log(f"  within20 med  : "
        f"{100*summary['median_per_scene']['frac_within_20cm']:.1f} %")
    log(f"  wrote {out/'summary.json'}")


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == '__main__':
    main()

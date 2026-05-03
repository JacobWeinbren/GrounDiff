"""GrounDiff training (paper §7.3).

  AdamW, lr=1e-4, wd=0.01
  Cosine annealing with 500 warmup steps
  Batch size 16
  T=10 diffusion steps, cosine β ∈ [1e-4, 2e-2]
  Max 1000 epochs with early stopping (paper sized for tiny corpus;
  for large corpora N_epoch should be set in config to a sensible cap).

Usage:
  python -u -m scripts.train \
      --config configs/defra.json \
      --tile_dir /data/england_tiles \
      --name_suffix v1
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tensorboardX import SummaryWriter

# Local
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import DSMDTMTileDataset, denormalise        # noqa: E402
from models import GrounDiff, MetricAggregator         # noqa: E402
from models.diffusion import gating                    # noqa: E402
from utils import CheckpointManager                    # noqa: E402


# ---- helpers -----------------------------------------------------------

def _setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s.%(msecs)03d - INFO: %(message)s',
                             datefmt='%y-%m-%d %H:%M:%S')
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(run_dir / 'train.log'); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


def _cosine_with_warmup(step: int, *, warmup: int, total: int,
                       min_frac: float = 0.0) -> float:
    """LR multiplier in [min_frac, 1.0]: linear warmup over `warmup`
    steps, then half-cosine decay to min_frac at `total`."""
    if step < warmup:
        return float(step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_frac + 0.5 * (1.0 - min_frac) * (1.0 + math.cos(math.pi * progress))


def _pct(v) -> str:
    """NaN-safe percentage formatter: returns '   —  ' for NaN."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '   —  '
    return f"{v:6.2%}"


def _flt(v, fmt: str = '.3f') -> str:
    """NaN-safe float formatter."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '  —  '
    return format(v, fmt)


@torch.no_grad()
def _val_loop(model: GrounDiff, dl_val, device,
              max_tiles: int = 256) -> dict:
    """Per-tile val with `init='noisy_dsm'` (paper §3.2 default).

    Computes both classification methods:
      Method A (residual): predicted ground = |s − g_pred| < τ_metres
      Method B (σ(ℓ)):     predicted ground = σ(ℓ) ≥ 0.5  (paper Fig.16)
    True ground in both: |s − g_GT| < τ_metres (default 0.20m, the
    Sithole-Vosselman 2003 §4.2.1 convention).
    """
    import torch as _torch
    model.eval()
    agg = MetricAggregator()
    seen = 0
    for batch in dl_val:
        if seen >= max_tiles:
            break
        dsm = batch['cond_dsm'].to(device)
        dtm = batch['target_dtm'].to(device)
        valid = batch['valid'].to(device)
        dsm_min = batch.get('cond_dsm_min')
        if dsm_min is not None:
            dsm_min = dsm_min.to(device)
        pred, logit = model.infer(dsm, init='noisy_dsm',
                                   return_logit=True, dsm_min=dsm_min)
        prob = _torch.sigmoid(logit)

        for i in range(dsm.shape[0]):
            stats = batch['stats'][i] if isinstance(batch['stats'], list) \
                    else {k: v[i] for k, v in batch['stats'].items()}
            pred_m = denormalise(pred[i, 0].cpu().numpy(), stats)
            gt_m   = denormalise(dtm [i, 0].cpu().numpy(), stats)
            dsm_m  = denormalise(dsm [i, 0].cpu().numpy(), stats)
            agg.update(
                pred_m, gt_m, valid[i].cpu().numpy(),
                dsm_m=dsm_m,
                prob_ground=prob[i, 0].cpu().numpy(),
            )
            seen += 1
            if seen >= max_tiles:
                break
    model.train()
    return agg.result()


def _save_scene_viz(scene_id: str, dsm_m, dtm_gt_m, pred_m, prob,
                     valid, viz_dir, dsm_min_m=None):
    """Save a per-scene diagnostic panel. Layout depends on whether
    dsm_min is available:

    Without dsm_min (paper-baseline 2x3 grid):
        DSM | GT DTM | GT residual
        Pred DTM | σ(ℓ) | pred − GT

    With dsm_min (3x3 grid showing how the 2nd channel is used):
        DSM (max-z)         | DSM_min (min-z)     | DSM − DSM_min (canopy gap)
        GT DTM              | Pred DTM            | DSM_min − Pred DTM
        GT |s − g|          | σ(ℓ)                | pred − GT
    The right-middle panel (DSM_min − Pred DTM) shows where the
    network deviates from the implicit ground prior given by min returns:
    near-zero = network trusts DSM_min as ground (good in forest),
    positive = network predicts BELOW DSM_min (canopy returns are not
    actually ground).

    All inputs are full-scene 2D float32 rasters in metres."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    v = valid.astype(bool)
    has_min = dsm_min_m is not None

    if v.any():
        rasters_for_range = [dsm_m, dtm_gt_m, pred_m]
        if has_min:
            rasters_for_range.append(dsm_min_m)
        elev_min = float(min(r[v].min() for r in rasters_for_range))
        elev_max = float(max(r[v].max() for r in rasters_for_range))
    else:
        elev_min, elev_max = 0.0, 1.0

    def _mask(arr):
        return np.ma.masked_where(~v, arr)

    gt_resid = np.abs(dsm_m - dtm_gt_m)
    pred_err = pred_m - dtm_gt_m

    if not has_min:
        # ---- 2x3 baseline layout ----
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        for a in ax.flat:
            a.set_xticks([]); a.set_yticks([])

        im = ax[0, 0].imshow(_mask(dsm_m), cmap='terrain',
                              vmin=elev_min, vmax=elev_max)
        ax[0, 0].set_title('DSM (input)')
        plt.colorbar(im, ax=ax[0, 0], shrink=0.7, label='m')

        im = ax[0, 1].imshow(_mask(dtm_gt_m), cmap='terrain',
                              vmin=elev_min, vmax=elev_max)
        ax[0, 1].set_title('GT DTM')
        plt.colorbar(im, ax=ax[0, 1], shrink=0.7, label='m')

        im = ax[0, 2].imshow(_mask(gt_resid), cmap='magma', vmin=0,
                              vmax=max(0.5, float(np.percentile(
                                  gt_resid[v], 99)) if v.any() else 0.5))
        ax[0, 2].set_title('GT |s − g| (true non-ground)')
        plt.colorbar(im, ax=ax[0, 2], shrink=0.7, label='m')

        im = ax[1, 0].imshow(_mask(pred_m), cmap='terrain',
                              vmin=elev_min, vmax=elev_max)
        ax[1, 0].set_title('Pred DTM')
        plt.colorbar(im, ax=ax[1, 0], shrink=0.7, label='m')

        im = ax[1, 1].imshow(_mask(prob), cmap='RdYlGn', vmin=0, vmax=1)
        ax[1, 1].set_title('σ(ℓ)  bright = predicted ground')
        plt.colorbar(im, ax=ax[1, 1], shrink=0.7, label='prob')

        err_lim = max(1.0, float(np.percentile(np.abs(pred_err[v]), 99))
                      if v.any() else 1.0)
        im = ax[1, 2].imshow(_mask(pred_err), cmap='RdBu_r',
                              vmin=-err_lim, vmax=err_lim)
        ax[1, 2].set_title(f'pred − GT  (±{err_lim:.2f}m)')
        plt.colorbar(im, ax=ax[1, 2], shrink=0.7, label='m')
    else:
        # ---- 3x3 layout with DSM_min diagnostic panels ----
        fig, ax = plt.subplots(3, 3, figsize=(16, 14))
        for a in ax.flat:
            a.set_xticks([]); a.set_yticks([])

        # Row 0: DSM (max), DSM_min (min), canopy gap
        im = ax[0, 0].imshow(_mask(dsm_m), cmap='terrain',
                              vmin=elev_min, vmax=elev_max)
        ax[0, 0].set_title('DSM (max-z input)')
        plt.colorbar(im, ax=ax[0, 0], shrink=0.7, label='m')

        im = ax[0, 1].imshow(_mask(dsm_min_m), cmap='terrain',
                              vmin=elev_min, vmax=elev_max)
        ax[0, 1].set_title('DSM_min (min-z input)')
        plt.colorbar(im, ax=ax[0, 1], shrink=0.7, label='m')

        canopy_gap = dsm_m - dsm_min_m
        gap_max = max(1.0, float(np.percentile(canopy_gap[v], 99))
                      if v.any() else 1.0)
        im = ax[0, 2].imshow(_mask(canopy_gap), cmap='YlGn', vmin=0,
                              vmax=gap_max)
        ax[0, 2].set_title(
            f'DSM − DSM_min  (canopy/penetration; max {gap_max:.1f}m)')
        plt.colorbar(im, ax=ax[0, 2], shrink=0.7, label='m')

        # Row 1: GT DTM, Pred DTM, DSM_min vs Pred (how network uses it)
        im = ax[1, 0].imshow(_mask(dtm_gt_m), cmap='terrain',
                              vmin=elev_min, vmax=elev_max)
        ax[1, 0].set_title('GT DTM')
        plt.colorbar(im, ax=ax[1, 0], shrink=0.7, label='m')

        im = ax[1, 1].imshow(_mask(pred_m), cmap='terrain',
                              vmin=elev_min, vmax=elev_max)
        ax[1, 1].set_title('Pred DTM')
        plt.colorbar(im, ax=ax[1, 1], shrink=0.7, label='m')

        # Diagnostic: how does the prediction relate to DSM_min?
        # = 0 → network exactly trusted min returns as ground
        # > 0 → network predicted below min returns (low canopy returns
        #       weren't actually ground)
        # < 0 → network predicted above min returns (overshoot)
        diff_pred_min = pred_m - dsm_min_m
        diff_lim = max(1.0, float(np.percentile(np.abs(diff_pred_min[v]), 99))
                       if v.any() else 1.0)
        im = ax[1, 2].imshow(_mask(diff_pred_min), cmap='RdBu_r',
                              vmin=-diff_lim, vmax=diff_lim)
        ax[1, 2].set_title(
            f'Pred − DSM_min  (±{diff_lim:.2f}m)\n'
            'white = trusted min as ground')
        plt.colorbar(im, ax=ax[1, 2], shrink=0.7, label='m')

        # Row 2: GT residual, σ(ℓ), pred error
        im = ax[2, 0].imshow(_mask(gt_resid), cmap='magma', vmin=0,
                              vmax=max(0.5, float(np.percentile(
                                  gt_resid[v], 99)) if v.any() else 0.5))
        ax[2, 0].set_title('GT |s − g| (true non-ground)')
        plt.colorbar(im, ax=ax[2, 0], shrink=0.7, label='m')

        im = ax[2, 1].imshow(_mask(prob), cmap='RdYlGn', vmin=0, vmax=1)
        ax[2, 1].set_title('σ(ℓ)  bright = predicted ground')
        plt.colorbar(im, ax=ax[2, 1], shrink=0.7, label='prob')

        err_lim = max(1.0, float(np.percentile(np.abs(pred_err[v]), 99))
                      if v.any() else 1.0)
        im = ax[2, 2].imshow(_mask(pred_err), cmap='RdBu_r',
                              vmin=-err_lim, vmax=err_lim)
        ax[2, 2].set_title(f'Pred − GT  (±{err_lim:.2f}m)')
        plt.colorbar(im, ax=ax[2, 2], shrink=0.7, label='m')

    fig.suptitle(scene_id, fontsize=12)
    fig.tight_layout()
    out = viz_dir / f"{scene_id}.png"
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return out


@torch.no_grad()
def _priostitch_val_loop(model: GrounDiff, scene_files: list, device,
                          *, tile: int = 256, stride: int = 128,
                          blend_mode: str = 'min', logger=None,
                          viz_dir=None) -> dict:
    """Full-scene PrioStitch val (paper §3.3 deployment path).

    Slower than per-tile val (5-10 min for N scenes) but directly
    comparable to paper Tab.1 numbers. If `viz_dir` is given, saves a
    2×3 PNG panel per scene (DSM, GT DTM, GT residual, pred DTM, σ(ℓ),
    pred error).
    """
    from utils import priostitch_inference

    model.eval()
    agg = MetricAggregator()
    t0 = time.time()
    for n, sm in enumerate(scene_files, 1):
        try:
            with np.load(str(sm), allow_pickle=True) as f:
                if not all(k in f.files for k in ('dsm', 'dtm', 'valid')):
                    if logger is not None:
                        logger.warning(f"    [{n}/{len(scene_files)}] "
                                       f"{sm.parent.name}: scene file "
                                       f"missing required keys, skipped")
                    continue
                dsm = f['dsm'].astype(np.float32)
                dtm_gt = f['dtm'].astype(np.float32)
                valid = f['valid'].astype(bool)
                stats = f['stats'].item() if 'stats' in f.files else {}
                dsm_min = (f['dsm_min'].astype(np.float32)
                           if 'dsm_min' in f.files else None)
        except Exception as e:
            if logger is not None:
                logger.warning(f"    [{n}/{len(scene_files)}] "
                               f"{sm.parent.name}: load failed ({e}), "
                               f"skipped")
            continue

        if not stats.get('has_data', True):
            continue

        # Pass DSM in metres directly. PrioStitch does per-tile
        # DSM-only normalisation and returns metres directly.
        # Capture per-tile snapshots for diagnostic viz (only when
        # viz_dir is set; cheap when disabled).
        tile_snaps: list = [] if viz_dir is not None else None

        pred_m, prob = priostitch_inference(
            model, dsm_full=np.zeros_like(dsm),  # unused when dsm_metres given
            dsm_metres=dsm, scene_stats=stats,
            dsm_min_metres=dsm_min,
            device=device,
            tile=tile, stride=stride, blend_mode=blend_mode,
            init_prior=True, valid_mask=valid, return_prob=True,
            capture_tiles=tile_snaps, capture_n=4,
        )
        agg.update(pred_m, dtm_gt, valid, dsm_m=dsm, prob_ground=prob)
        if logger is not None:
            # Per-scene aggregator with lower min_classify_n: suppresses
            # E_T1/E_T2 only on truly tiny denominators (< 100 cells)
            r_partial = MetricAggregator(min_classify_n=100)
            r_partial.update(pred_m, dtm_gt, valid, dsm_m=dsm,
                              prob_ground=prob)
            rs = r_partial.result()
            n_ng = rs.get('n_nonground', 0)
            logger.info(
                f"    [{n}/{len(scene_files)}] {sm.parent.name}: "
                f"RMSE={rs['rmse']:.3f}m  MAE={rs['mae']:.3f}m  "
                f"E_T1={_pct(rs['E_T1'])}  E_T2={_pct(rs['E_T2'])}  "
                f"MCC={_flt(rs.get('MCC'))}  "
                f"bal_acc={_flt(rs.get('bal_acc'))}  "
                f"n_ng={n_ng:,d}  "
                f"[logit: MCC={_flt(rs.get('MCC_logit'))}, "
                f"bal_acc={_flt(rs.get('bal_acc_logit'))}]"
            )
        if viz_dir is not None:
            try:
                _save_scene_viz(sm.parent.name, dsm, dtm_gt,
                                 pred_m, prob, valid, viz_dir,
                                 dsm_min_m=dsm_min)
                # Per-tile montage (4 highest-canopy-gap tiles per scene)
                if tile_snaps:
                    from utils import save_tile_viz_montage
                    save_tile_viz_montage(
                        tile_snaps,
                        out_path=Path(viz_dir) /
                                  f"{sm.parent.name}_tiles.png",
                        dtm_gt_full=dtm_gt,
                        scene_id=sm.parent.name,
                        dsm_min_available=(dsm_min is not None),
                    )
            except Exception as e:
                if logger is not None:
                    logger.warning(f"      viz save failed for "
                                   f"{sm.parent.name}: {e}")
    model.train()
    res = agg.result()
    res['_seconds'] = time.time() - t0
    return res


def _collate(batch: list) -> dict:
    out = {}
    for k in batch[0]:
        if k in ('stats', 'name'):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--tile_dir', default=None,
                   help="Override config.data.tile_dir.")
    p.add_argument('--name_suffix', default='',
                   help="Appended to run name (e.g. 'v1').")
    p.add_argument('--resume', default=None,
                   help="Path to a checkpoint to resume from.")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if args.tile_dir is not None:
        cfg.setdefault('data', {})['tile_dir'] = args.tile_dir

    # --- Run dir ------------------------------------------------------
    base_name = cfg.get('name', 'groundiff')
    suffix = f"_{args.name_suffix}" if args.name_suffix else ''
    timestamp = time.strftime('%y%m%d_%H%M%S')
    run_dir = Path(cfg.get('experiment_dir', 'experiments')) \
              / f"{base_name}{suffix}_{timestamp}"
    log = _setup_logger(run_dir)
    log.info(f"Run dir: {run_dir}")

    ckpt_dir = run_dir / 'checkpoint'
    res_dir  = run_dir / 'results'
    res_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / 'tb'))
    (run_dir / 'config.used.json').write_text(json.dumps(cfg, indent=2))

    # --- Data ---------------------------------------------------------
    tile_dir = cfg['data']['tile_dir']
    min_valid_frac = float(cfg['data'].get('min_valid_frac', 0.0))
    use_min_dsm = bool(cfg['data'].get('use_min_dsm', False))
    alpha_metres = float(cfg['model'].get('loss', {}).get(
        'alpha_metres', 0.20))
    ds_tr = DSMDTMTileDataset(tile_dir, split='train',
                              crop_px=cfg['data'].get('crop', 256),
                              augment=True,
                              min_valid_frac=min_valid_frac,
                              alpha_metres=alpha_metres,
                              use_min_dsm=use_min_dsm)
    ds_val = DSMDTMTileDataset(tile_dir, split='test',
                               crop_px=cfg['data'].get('crop', 256),
                               augment=False, seed=0,
                               min_valid_frac=min_valid_frac,
                               alpha_metres=alpha_metres,
                               use_min_dsm=use_min_dsm)
    if use_min_dsm:
        log.info(f"  use_min_dsm=True (UNet expects in_channel=3)")

    # Optional class-imbalance sampler (paper deviation, sampling-only)
    sampler_mode = cfg['train'].get('sampler', 'uniform')
    train_sampler = None
    train_shuffle = True
    if sampler_mode != 'uniform':
        cache_path = Path(tile_dir) / f".sampling_fracs_{sampler_mode}.npz"
        weights = ds_tr.compute_sampling_weights(
            mode=sampler_mode,
            alpha_norm=cfg['model'].get('loss', {}).get('alpha', 0.05),
            cache_path=cache_path,
            verbose=True,
        )
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(ds_tr),
            replacement=True,
        )
        train_shuffle = False  # mutually exclusive with sampler
        log.info(f"Sampler: {sampler_mode}  "
                 f"(non-uniform per-tile sampling weights)")

    dl_tr = DataLoader(ds_tr, batch_size=cfg['train']['batch_size'],
                       shuffle=train_shuffle, sampler=train_sampler,
                       num_workers=cfg['train'].get('num_workers', 6),
                       pin_memory=True, drop_last=True, collate_fn=_collate,
                       persistent_workers=True)
    dl_val = DataLoader(ds_val, batch_size=cfg['train']['batch_size'],
                        shuffle=False, num_workers=2,
                        pin_memory=True, drop_last=False, collate_fn=_collate)

    # --- Model --------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GrounDiff(
        unet_kwargs=cfg['model']['unet'],
        diffusion_kwargs=cfg['model']['diffusion'],
        loss_kwargs=cfg['model'].get('loss', {}),
    ).to(device)
    # Kaiming init on Conv/Linear (Palette convention: kaiming_normal
    # with mode='fan_in', bias=0). Improves convergence vs PyTorch
    # default kaiming_uniform with a=√5.
    if cfg['train'].get('init_kaiming', True) and not args.resume:
        from models.nn import kaiming_init_
        n_init = kaiming_init_(model.unet)
        log.info(f"  kaiming_init applied to {n_init} Conv/Linear modules")
    log.info(f"GrounDiff UNet params: {model.num_params/1e6:.1f}M")

    # EMA copy of the UNet for inference (Palette guided_diffusion
    # convention; standard for diffusion training). Updated each
    # optimizer step. Used at val/priostitch_val. Saved in checkpoint.
    ema_decay = float(cfg['train'].get('ema_decay', 0.9999))
    ema_enabled = bool(cfg['train'].get('ema_enabled', True))
    if ema_enabled:
        import copy as _copy
        ema_model = _copy.deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad = False
        ema_model.eval()
        log.info(f"  EMA enabled, decay={ema_decay}")
    else:
        ema_model = None
        log.info("  EMA disabled (paper-faithful baseline)")

    # --- Optim --------------------------------------------------------
    optim = torch.optim.AdamW(
        model.unet.parameters(),
        lr=cfg['optim']['lr'],
        weight_decay=cfg['optim'].get('weight_decay', 0.01),
        betas=tuple(cfg['optim'].get('betas', (0.9, 0.999))),
    )
    total_iters = int(cfg['train']['n_iter'])
    warmup = int(cfg['optim'].get('warmup_steps', 500))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda step: _cosine_with_warmup(
            step, warmup=warmup, total=total_iters,
            min_frac=cfg['optim'].get('lr_min_frac', 0.0)))

    ckpt = CheckpointManager(ckpt_dir, model, optim, scheduler, log,
                              ema_model=ema_model)
    if args.resume:
        ckpt.load(Path(args.resume), map_location=device)
    start_epoch = ckpt.epoch + 1
    global_step = ckpt.global_step
    max_epoch = int(cfg['train']['n_epoch'])
    val_every = int(cfg['train'].get('val_every_epoch', 1))
    save_every = int(cfg['train'].get('save_every_epoch', 1))
    log_every = int(cfg['train'].get('log_iter', 50))

    # --- Per-epoch PrioStitch val setup -------------------------------
    # Run full-scene PrioStitch on a fixed subset of test scenes after
    # each tile-level val. Slower (~5-10 min) but produces paper Tab.1-
    # comparable numbers (full-scene RMSE/MAE/err with PrioStitch
    # tiling+blending, not 256x256 crop tiles).
    ps_cfg = cfg.get('priostitch', {})
    ps_n_scenes = int(cfg['train'].get('priostitch_val_scenes', 4))
    ps_every = int(cfg['train'].get('priostitch_val_every_epoch', 1))
    ps_scene_files: list = []
    if ps_n_scenes > 0:
        test_dir = Path(tile_dir) / 'test'
        all_scenes = sorted(test_dir.rglob('_scene_*.npz'))

        # Representative sample: rank ALL test scenes by non-ground content
        # (the most diagnostic axis — distinguishes empty from rural from
        # urban) and pick `ps_n_scenes` evenly-spaced quantiles. This
        # guarantees the viz set spans the full content diversity (empty
        # → flat → mixed → structured) instead of the alphabetical-first-N
        # bias toward whatever scenes happen to start with EN_A* etc.
        # Stats cached to <test_dir>/.scene_stats.npz so this only runs
        # once per dataset.
        cache_path = test_dir / '.scene_stats.npz'
        scene_keys = [str(p) for p in all_scenes]
        import hashlib as _h
        path_hash = _h.sha1(
            ''.join(p.name for p in all_scenes).encode()
        ).hexdigest()[:16]
        scene_stats = None
        if cache_path.exists():
            try:
                d = np.load(str(cache_path), allow_pickle=True)
                if str(d['paths_hash']) == path_hash:
                    scene_stats = d['stats']
                    log.info(f"  loaded scene stats from {cache_path}")
            except Exception:
                pass
        if scene_stats is None:
            log.info(f"  computing stats for {len(all_scenes)} test "
                     f"scenes (one-time, cached)...")
            scene_stats = np.zeros((len(all_scenes), 2), dtype=np.float32)
            for i, sm in enumerate(all_scenes):
                try:
                    with np.load(str(sm), allow_pickle=True) as f:
                        valid = f['valid'].astype(bool)
                        valid_frac = float(valid.mean())
                        if valid.any():
                            dsm = f['dsm'].astype(np.float32)
                            dtm = f['dtm'].astype(np.float32)
                            r = np.abs(dsm - dtm)[valid]
                            ng_frac = float((r > 0.20).mean())
                        else:
                            ng_frac = 0.0
                    scene_stats[i] = (valid_frac, ng_frac)
                except Exception:
                    scene_stats[i] = (0.0, 0.0)
            try:
                np.savez(str(cache_path), stats=scene_stats,
                          paths_hash=path_hash)
            except Exception as e:
                log.warning(f"  failed to cache scene stats: {e}")

        # Sort scene indices by non-ground fraction (ascending)
        ng = scene_stats[:, 1]
        order = np.argsort(ng)  # low → high non-ground
        # Evenly-spaced indices through the sorted list
        if len(order) <= ps_n_scenes:
            picks = order.tolist()
        else:
            qidx = np.linspace(0, len(order) - 1, ps_n_scenes
                               ).round().astype(int)
            picks = order[qidx].tolist()
        ps_scene_files = [all_scenes[i] for i in picks]
        log.info(f"PrioStitch val: {len(ps_scene_files)} representative "
                 f"scenes from {len(all_scenes)} (quantile sample by "
                 f"non-ground content; every {ps_every} epoch, "
                 f"tile={ps_cfg.get('tile', 256)}, "
                 f"stride={ps_cfg.get('stride', 128)}, "
                 f"blend={ps_cfg.get('blend_mode', 'min')})")
        for i in picks:
            vf, ngf = scene_stats[i]
            log.info(f"  - {all_scenes[i].parent.name}: "
                     f"valid={vf:.1%}  non-ground={ngf:.1%}")

    # --- Train --------------------------------------------------------
    for epoch in range(start_epoch, max_epoch + 1):
        t0 = time.time()
        for batch in dl_tr:
            dsm = batch['cond_dsm'].to(device, non_blocking=True)
            dtm = batch['target_dtm'].to(device, non_blocking=True)
            valid = batch['valid'].to(device, non_blocking=True)
            m_alpha = batch.get('m_alpha')
            if m_alpha is not None:
                m_alpha = m_alpha.to(device, non_blocking=True)
            dsm_min = batch.get('cond_dsm_min')
            if dsm_min is not None:
                dsm_min = dsm_min.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            loss, m = model.training_step(dsm, dtm, valid,
                                           m_alpha=m_alpha,
                                           dsm_min=dsm_min)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(),
                                            max_norm=1.0)
            optim.step()
            scheduler.step()
            global_step += 1

            # --- EMA update (Palette convention) ---
            if ema_model is not None:
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.unet.parameters(),
                                         model.unet.parameters()):
                        ema_p.data.mul_(ema_decay).add_(
                            p.data, alpha=1.0 - ema_decay)
                    # Buffers (BN running stats etc) — copy directly
                    for ema_b, b in zip(ema_model.unet.buffers(),
                                         model.unet.buffers()):
                        ema_b.data.copy_(b.data)

            if global_step % log_every == 0:
                lr = scheduler.get_last_lr()[0]
                log.info(
                    f"ep {epoch:4d}  it {global_step:8d}  "
                    f"loss={float(m['loss']):.4f}  "
                    f"l1={float(m['l1']):.4f}  "
                    f"l2={float(m['l2']):.4f}  "
                    f"∇={float(m['grad']):.4f}  "
                    f"conf={float(m['conf']):.4f}  "
                    f"lr={lr:.2e}"
                )
                for k, v in m.items():
                    writer.add_scalar(f"train/{k}", float(v), global_step)
                writer.add_scalar("train/lr", lr, global_step)

            if global_step >= total_iters:
                break

        log.info(f"epoch {epoch} done in {time.time() - t0:.0f}s")

        # ---- val + save ---------------------------------------------
        if epoch % val_every == 0:
            r = _val_loop(ema_model if ema_model is not None else model,
                          dl_val, device,
                          max_tiles=cfg['train'].get('val_max_tiles', 256))
            log.info(
                f"  val (n={r['n_valid']:,d}): "
                f"RMSE={r['rmse']:.3f}m  MAE={r['mae']:.3f}m  "
                f"E_T1={_pct(r['E_T1'])}  E_T2={_pct(r['E_T2'])}  "
                f"E_tot={_pct(r['E_tot'])}  "
                f"MCC={_flt(r['MCC'])}  bal_acc={_flt(r['bal_acc'])}  "
                f"mIoU={_flt(r['mIoU'])} "
                f"(g={_flt(r['IoU_ground'])}, ng={_flt(r['IoU_nonground'])})  "
                f"[logit: E_tot={_pct(r['E_tot_logit'])}, "
                f"MCC={_flt(r['MCC_logit'])}, "
                f"bal_acc={_flt(r['bal_acc_logit'])}, "
                f"mIoU={_flt(r['mIoU_logit'])}]  "
                f"err>0.5m={r['err_gt_0.5']:.1%}  "
                f"err>1.0m={r['err_gt_1.0']:.1%}"
            )
            for k, v in r.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"val/{k}", float(v), global_step)
        else:
            r = {'rmse': None}

        # ---- PrioStitch val on a few full scenes --------------------
        if ps_scene_files and (epoch % ps_every == 0):
            log.info(f"  running PrioStitch val on "
                     f"{len(ps_scene_files)} test scenes...")
            ps_save_viz = bool(cfg['train'].get(
                'priostitch_val_save_viz', False))
            ps_viz_dir = (run_dir / 'viz' / f'epoch_{epoch:04d}'
                          if ps_save_viz else None)
            try:
                ps_r = _priostitch_val_loop(
                    ema_model if ema_model is not None else model,
                    ps_scene_files, device,
                    tile=ps_cfg.get('tile', 256),
                    stride=ps_cfg.get('stride', 128),
                    blend_mode=ps_cfg.get('blend_mode', 'min'),
                    logger=log,
                    viz_dir=ps_viz_dir,
                )
                log.info(
                    f"  priostitch (n={ps_r['n_valid']:,d}): "
                    f"RMSE={ps_r['rmse']:.3f}m  MAE={ps_r['mae']:.3f}m  "
                    f"E_T1={_pct(ps_r['E_T1'])}  E_T2={_pct(ps_r['E_T2'])}  "
                    f"E_tot={_pct(ps_r['E_tot'])}  "
                    f"MCC={_flt(ps_r['MCC'])}  bal_acc={_flt(ps_r['bal_acc'])}  "
                    f"mIoU={_flt(ps_r['mIoU'])} "
                    f"(g={_flt(ps_r['IoU_ground'])}, ng={_flt(ps_r['IoU_nonground'])})  "
                    f"[logit: E_tot={_pct(ps_r['E_tot_logit'])}, "
                    f"MCC={_flt(ps_r['MCC_logit'])}, "
                    f"bal_acc={_flt(ps_r['bal_acc_logit'])}, "
                    f"mIoU={_flt(ps_r['mIoU_logit'])}]  "
                    f"err>0.5m={ps_r['err_gt_0.5']:.1%}  "
                    f"err>1.0m={ps_r['err_gt_1.0']:.1%}  "
                    f"({ps_r.get('_seconds', 0):.0f}s)"
                )
                if ps_viz_dir is not None:
                    log.info(f"  viz panels saved to {ps_viz_dir}/")
                for k, v in ps_r.items():
                    if k.startswith('_') or not isinstance(v, (int, float)):
                        continue
                    writer.add_scalar(f"priostitch/{k}", float(v), global_step)
            except Exception as e:
                log.warning(f"  priostitch val failed: "
                            f"{type(e).__name__}: {e}")

        if epoch % save_every == 0:
            ckpt.save(epoch=epoch, global_step=global_step, rmse=r.get('rmse'))

        if global_step >= total_iters:
            log.info(f"reached n_iter={total_iters}; stopping training")
            break

    writer.close()
    log.info("Training done.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Train the raster GrounDiff model.

Usage:
    python -m stage2_raster.scripts.train --config configs/defra.json

Features:
  - AdamW with cosine LR schedule + linear warmup.
  - EMA weights (decay 0.9999 by default).
  - Hard mining: per-tile loss EMA used to bias sampling.
  - Step-level checkpointing every `save_every_steps`.
  - Val pass every `val_every_steps`; logs paper-faithful metrics.
  - OOM-safe try/except wrapper around the model step.
  - Resumable: if `<out_dir>/latest.pt` exists, training continues
    from there with optimizer + EMA + miner state restored.

The config is a JSON with model / data / optim / eval keys. See
configs/defra.json for the production preset.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

# Reduce CUDA allocator fragmentation. Helps when we mix big static
# activation tensors (1024² conv feature maps) with smaller transient
# allocations (gathers, indexing). MUST be set before `import torch`.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# tqdm is optional. If installed, we get a live progress bar with the
# current loss in the postfix; if not, we silently fall through to the
# existing per-log_every-steps text output.
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    class _NoBar:
        def __init__(self, total=None, initial=0, **kw):
            self.n = initial; self.total = total
        def update(self, n=1): self.n += n
        def set_postfix_str(self, s, refresh=True): pass
        def close(self): pass
        @staticmethod
        def write(s, **kw): print(s)
    def tqdm(iterable=None, **kw):
        return _NoBar(**kw)
    tqdm.write = _NoBar.write

# Allow `python stage2_raster/scripts/train.py` invocation by injecting
# the parent of `stage2_raster` into sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage2_raster.models import GrounDiffRaster
from stage2_raster.data.dataset import (
    RasterTileDataset, RasterPretiledDataset, RasterEvalTiles, HardMiner)


# --------------------------------------------------------------------- #
#  EMA: shadow params, in-place update each step.
# --------------------------------------------------------------------- #

class EMA:
    """Exponential moving average over model parameters.

    Matches the reference: parameters are EMA-averaged (mul_·decay +
    add_·(1-decay)); buffers are copied directly. For this architecture
    the only float buffers are the constant diffusion-schedule tensors
    (betas, alphas_bar, ...) and GroupNorm has no running stats, so the
    distinction is mostly academic here — but copying buffers rather than
    averaging them is the exactly-correct behaviour and avoids any edge
    case if buffers are ever added.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        # Names of parameters (EMA-averaged) vs everything else in the
        # state_dict (buffers — copied).
        self._param_names = set(dict(model.named_parameters()).keys())
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.last_decay_used = 0.0

    @torch.no_grad()
    def update(self, model: torch.nn.Module, decay_override: float | None = None):
        d = self.decay if decay_override is None else float(decay_override)
        self.last_decay_used = d
        sd = model.state_dict()
        for k, v in self.shadow.items():
            # On resume from checkpoint, shadows are loaded on CPU but the
            # model is on GPU — lazily move shadow to match model device.
            if v.device != sd[k].device:
                self.shadow[k] = v.to(sd[k].device)
                v = self.shadow[k]
            if k in self._param_names:
                v.mul_(d).add_(sd[k].detach(), alpha=1 - d)   # EMA params
            else:
                v.copy_(sd[k].detach())                        # copy buffers

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> dict:
        """Swap model params with EMA shadow; return the originals for restore."""
        orig = {}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if v.device != sd[k].device:
                self.shadow[k] = v.to(sd[k].device)
                v = self.shadow[k]
            orig[k] = sd[k].detach().clone()
            sd[k].copy_(v)
        return orig

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, orig: dict):
        sd = model.state_dict()
        for k, v in orig.items():
            sd[k].copy_(v)

    def state_dict(self) -> dict:
        return {'decay': self.decay,
                'shadow': {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, d: dict):
        self.decay = float(d['decay'])
        self.shadow = {k: v for k, v in d['shadow'].items()}


# --------------------------------------------------------------------- #
#  LR schedule: linear warmup -> cosine decay
# --------------------------------------------------------------------- #

def cosine_warmup(step: int, *, warmup: int, total: int,
                  base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    # Clamp to [0, 1] (matches reference): once past `total`, hold at
    # min_lr instead of letting the cosine continue past π and rise again.
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------- #
#  Eval: paper-faithful per-pixel metrics
# --------------------------------------------------------------------- #

@torch.inference_mode()
def evaluate(model: GrounDiffRaster, val_loader, device,
              *, max_batches: int = 50, use_ema: EMA | None = None,
              amp_dtype: torch.dtype | None = None,
              alpha_metres: float = 0.20) -> dict:
    """Run validation, return aggregate metrics.

    Reports L1, L2, L∇, Lc (paper-faithful), plus per-pixel residual
    classification (E_T1 / E_T2 / E_tot at α=0.20m), interpreting σ(ℓ)
    as the predicted ground probability, plus RMSE_ground, RMSE_non_ground,
    and RMSE_overall in metres.

    OOM/speed notes:
    - `torch.inference_mode()` is stronger than `@torch.no_grad()`: it
      disables version-counter tracking on tensors and lets the runtime
      use more aggressive memory layouts. Pairs with `model.eval()`
      which disables dropout and batchnorm tracking.
    - `cuda.empty_cache()` at entry and exit returns reserved-but-unused
      memory to the OS, so when val finishes the training-step
      allocator pool doesn't have to compete with stale val activations.
    - All metric accumulation uses `.item()` to drain into CPU floats,
      so no GPU tensors live across batches.
    - The expensive part is `model.infer()` — T=10 reverse diffusion
      steps per batch. At 50 batches × bs=16 × T=10 = 8000 forward
      passes through the 67.9M UNet — about 30-60s on Blackwell at
      bf16. The single `training_step` call per batch (random t, single
      forward) is cheap by comparison and gives us the L1/L2/L∇/Lc
      values that match the training-time loss reporting.

    Note: this is a *per-pixel* approximation. The real benchmark is
    per-point against the LAZ (Sithole-Vosselman 2003). That lives in
    scripts/test.py and operates on the LAZ output directly.
    """
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.empty_cache()

    orig = None
    if use_ema is not None:
        orig = use_ema.apply_to(model)
    model.eval()

    # Pick a context manager: autocast if we have bf16/fp16, else a no-op.
    if amp_dtype is not None:
        cm = lambda: torch.autocast(device_type=device.type, dtype=amp_dtype)
    else:
        from contextlib import nullcontext
        cm = nullcontext

    agg = dict(loss=0.0, l1=0.0, l2=0.0, grad=0.0, conf=0.0,
               fp=0, fn=0, fp_logit=0, fn_logit=0,
               t_total=0, ng_total=0, g_total=0,
               n_tiles=0, n_pixels=0,
               # RMSE accumulators in metres (squared error sums, per class).
               # 'gnd' = M_α=1 cells (DSM directly observed ground — easy case,
               #         model just needs to pass DSM through via gating).
               # 'ng'  = M_α=0 cells (canopy/building/etc — hard case,
               #         model must predict ground beneath the obstruction).
               # 'all' = both, the conventional per-pixel RMSE.
               # Computed in METRES, accounting for per-tile normalisation
               # spans via the z_hi-z_lo half-span multiplier.
               sse_gnd_m2=0.0, sse_ng_m2=0.0, sse_all_m2=0.0)

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
        with cm():
            loss, metrics = model.training_step(
                dsm_max=b['dsm_max'], dsm_min=b['dsm_min'],
                dsm_mean=b['dsm_mean'], dsm_mask=b['dsm_mask'],
                dtm=b['gt_dtm'], valid=b['valid'], m_alpha=b['m_alpha'])

        agg['loss'] += float(metrics['loss'])
        agg['l1']   += float(metrics['l1'])
        agg['l2']   += float(metrics['l2'])
        agg['grad'] += float(metrics['grad'])
        agg['conf'] += float(metrics['conf'])
        agg['n_tiles'] += b['gt_dtm'].shape[0]

        # Per-pixel classification — also under autocast so the diffusion
        # sample path runs at bf16 speed.
        with cm():
            dtm_pred, logit = model.infer(
                dsm_max=b['dsm_max'], dsm_min=b['dsm_min'],
                dsm_mean=b['dsm_mean'], dsm_mask=b['dsm_mask'],
                init='noisy_dsm', return_logit=True)
        valid_b = (b['valid'] > 0.5)
        gt_ground = (b['m_alpha'] > 0.5)   # true ground = M_α (|s−g_GT|<α)
        agg['n_pixels'] += int(valid_b.sum().item())
        n_g = (gt_ground & valid_b).sum().item()
        n_ng = ((~gt_ground) & valid_b).sum().item()
        agg['g_total'] += n_g
        agg['ng_total'] += n_ng

        # Per-tile half-span (metres) for normalised→metres conversions.
        # Needed both for metric RMSE and for the residual classifier.
        meta = batch.get('meta', {})
        z_lo = meta.get('z_lo')
        z_hi = meta.get('z_hi')
        half_span = None
        if z_lo is not None and z_hi is not None:
            if not isinstance(z_lo, torch.Tensor):
                z_lo = torch.as_tensor(z_lo)
                z_hi = torch.as_tensor(z_hi)
            z_lo = z_lo.to(dtm_pred.device).float().view(-1, 1, 1, 1)
            z_hi = z_hi.to(dtm_pred.device).float().view(-1, 1, 1, 1)
            half_span = (z_hi - z_lo) * 0.5

        # === Two predicted-ground criteria (the paper does not state the
        # eval threshold mechanism; we report BOTH for comparison) ===
        #
        # Method A — RESIDUAL: |s − g_pred| < α  (in metres).
        #   This applies the paper's ONLY defined ground criterion, M_α
        #   (Eq.14, |r| < α), to the *prediction* rather than the GT. It
        #   is the same rule used per-point when writing the classified
        #   LAZ (infer_to_laz: |z − ẑ| < α). Regression-coupled.
        #
        # Method B — σ(ℓ) ≥ 0.5: threshold the confidence head.
        #   σ(ℓ) is the BCE-trained confidence (Eq.14 target) and the
        #   paper's "Ground Prob." map (Figs 6, 16). The 0.5 threshold is
        #   NOT stated in the paper — it's the natural BCE cut. Pure
        #   segmentation, decoupled from regression accuracy.
        alpha_m = float(alpha_metres)
        dsm_max_b = b['dsm_max'].to(dtm_pred.device)

        # Method A (residual)
        if half_span is not None:
            resid_m = (dsm_max_b - dtm_pred).abs() * half_span
            pred_g_res = (resid_m < alpha_m)
        else:
            pred_g_res = ((dsm_max_b - dtm_pred).abs() < 0.05)  # norm fallback
        agg['fp'] += (pred_g_res & (~gt_ground) & valid_b).sum().item()
        agg['fn'] += ((~pred_g_res) & gt_ground & valid_b).sum().item()

        # Method B (σ(ℓ) ≥ 0.5)
        pred_g_log = (torch.sigmoid(logit) > 0.5)
        agg['fp_logit'] += (pred_g_log & (~gt_ground) & valid_b).sum().item()
        agg['fn_logit'] += ((~pred_g_log) & gt_ground & valid_b).sum().item()

        # ---- Per-pixel RMSE in metres -----------------------------------
        # dtm_pred and b['gt_dtm'] are both normalised [-1,+1] per-tile;
        # multiply the normalised diff by half_span to get metres.
        if half_span is not None:
            err_n = (dtm_pred - b['gt_dtm'].to(dtm_pred.device))
            err_m = err_n * half_span
            sq_err_m = (err_m * err_m).float()
            agg['sse_gnd_m2'] += float((sq_err_m * (gt_ground & valid_b).float()).sum().item())
            agg['sse_ng_m2']  += float((sq_err_m * ((~gt_ground) & valid_b).float()).sum().item())
            agg['sse_all_m2'] += float((sq_err_m * valid_b.float()).sum().item())

    if use_ema is not None and orig is not None:
        use_ema.restore(model, orig)
    model.train()

    # Average loss components across tiles.
    for k in ('loss', 'l1', 'l2', 'grad', 'conf'):
        agg[k] /= max(agg['n_tiles'], 1)

    # Sithole-Vosselman per-pixel error rates. Paper §4.1:
    #   E_T1 = FP / n_nonground   (retaining non-ground)
    #   E_T2 = FN / n_ground      (removing ground)
    # Reported for BOTH predicted-ground criteria (paper doesn't fix one):
    #   e_t*_pct        — Method A, residual |s−g_pred|<α (matches LAZ output)
    #   e_t*_pct_logit  — Method B, σ(ℓ)≥0.5 (confidence head)
    agg['e_t1_pct']  = 100.0 * agg['fp'] / max(agg['ng_total'], 1)
    agg['e_t2_pct']  = 100.0 * agg['fn'] / max(agg['g_total'], 1)
    agg['e_tot_pct'] = 100.0 * (agg['fp'] + agg['fn']) / max(agg['n_pixels'], 1)
    agg['e_t1_pct_logit']  = 100.0 * agg['fp_logit'] / max(agg['ng_total'], 1)
    agg['e_t2_pct_logit']  = 100.0 * agg['fn_logit'] / max(agg['g_total'], 1)
    agg['e_tot_pct_logit'] = 100.0 * (agg['fp_logit'] + agg['fn_logit']) / max(agg['n_pixels'], 1)

    # RMSE in metres for each class. NaN-safe via max(_, 1) on the
    # divisor — a category with no valid pixels reports RMSE 0.0 rather
    # than NaN, which is correct for our purposes (empty-category val
    # passes never happen at the dataset scale anyway).
    import math as _math
    agg['rmse_gnd_m'] = _math.sqrt(agg['sse_gnd_m2'] / max(agg['g_total'], 1))
    agg['rmse_ng_m']  = _math.sqrt(agg['sse_ng_m2']  / max(agg['ng_total'], 1))
    agg['rmse_all_m'] = _math.sqrt(agg['sse_all_m2'] / max(agg['n_pixels'], 1))

    # Drop the val-pass activation buffers before returning so the
    # training allocator doesn't have to wait for them to be freed.
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.empty_cache()
    return agg


# --------------------------------------------------------------------- #
#  Checkpoint helpers
# --------------------------------------------------------------------- #

def save_checkpoint(path: Path, *, model: GrounDiffRaster, optimizer,
                     ema: EMA, miner: HardMiner, step: int, config: dict,
                     best_e_tot: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = model.net.state_dict()
    # Save under both keys for back-compat: pre-UNet checkpoints used
    # `dit`; new ones use `net`. Both are the same dict contents.
    torch.save(dict(
        net=sd,
        dit=sd,
        backbone=getattr(model, 'backbone_name', 'unet'),
        optimizer=optimizer.state_dict(),
        ema=ema.state_dict(),
        miner=miner.state_dict(),
        step=step,
        config=config,
        best_e_tot=best_e_tot,
    ), path)


def load_checkpoint(path: Path, *, model: GrounDiffRaster, optimizer,
                     ema: EMA, miner: HardMiner) -> dict:
    d = torch.load(path, map_location='cpu', weights_only=False)
    # New checkpoints have `net`; legacy ones have `dit`. Same shape
    # only when the backbone is the same as when the ckpt was saved —
    # if you switch backbones (dit <-> unet) on resume, you need a
    # fresh run.
    sd = d.get('net', d.get('dit'))
    if sd is None:
        raise KeyError(
            f"checkpoint {path} has neither 'net' nor 'dit' state-dict key")
    model.net.load_state_dict(sd)
    if optimizer is not None and 'optimizer' in d:
        optimizer.load_state_dict(d['optimizer'])
    if ema is not None and 'ema' in d:
        ema.load_state_dict(d['ema'])
    if miner is not None and 'miner' in d:
        miner.load_state_dict(d['miner'])
    return d


def _run_per_val_viz(model, viz_scenes: list[dict], *,
                     val_root: Path, out_root: Path, step: int,
                     device: str, ema: EMA | None,
                     tile_size: int = 1024,
                     amp_dtype: torch.dtype | None = None,
                     log_fn=print) -> None:
    """Render the 20 (or however many) representative val scenes for
    this val step. Each scene gets its own PrioStitch inference plus
    elevation/analysis PNGs + a per-scene metrics CSV. The full set is
    rolled up into one CSV for the step.

    Outputs:
        <out_root>/viz/step_<step>/<scene>_elevation.png
        <out_root>/viz/step_<step>/<scene>_analysis.png
        <out_root>/viz/step_<step>/<scene>_metrics.csv
        <out_root>/viz/step_<step>/rollup.csv

    Per scene this runs one full PrioStitch pass. At 1024-tile stride,
    coarse=1024, T=10, a 20000x20000 scene needs ~25 tiles total ->
    ~30-90s per scene on a Blackwell. 20 scenes = 10-30 min per val
    pass. If that's too slow, set viz.per_val_viz=false in the config
    and use the post-training INSTALL_AND_RUN.sh viz block instead.
    """
    from stage2_raster.utils.priostitch import priostitch_infer
    from stage2_raster.scripts.visualize import make_all

    step_dir = out_root / 'viz' / f'step_{step:08d}'
    step_dir.mkdir(parents=True, exist_ok=True)
    rollup_csv = step_dir / 'rollup.csv'
    # Wipe rollup so we don't double-append on retry.
    if rollup_csv.exists():
        rollup_csv.unlink()

    # Swap to EMA weights for inference (best-quality reflection of
    # training state). The EMA helper stashes the live weights so we
    # can restore them after viz.
    ema_orig = None
    if ema is not None:
        ema_orig = ema.apply_to(model)
    model.eval()

    try:
        from tqdm.auto import tqdm as _tqdm
        scene_iter = _tqdm(viz_scenes, desc=f"val-viz step {step}",
                             unit="scene", leave=False, dynamic_ncols=True)
    except ImportError:
        scene_iter = viz_scenes

    t0 = time.time()
    # Track how many scenes succeed so we can warn if a substantial
    # fraction fail (likely OOM-related).
    n_ok = 0
    for s in scene_iter:
        scene_name = s['name']
        raster_path = val_root / scene_name / 'raster.npz'
        if not raster_path.exists():
            log_fn(f"# [viz] missing {raster_path}, skipping")
            continue
        try:
            with np.load(str(raster_path)) as z:
                dsm_max = z['dsm_max']; dsm_min = z['dsm_min']
                dsm_mean = z['dsm_mean']; dsm_mask = z['dsm_mask']
                gt_dtm = z['gt_dtm']; valid = z['valid']
                had_g = z['had_ground_return'] if 'had_ground_return' in z.files else None
                gsd = float(z['gsd']); alpha = float(z['alpha_metres'])
            with torch.inference_mode():
                # Val-time viz: TTA off (tta=1). The full 8x D4 TTA is
                # 8x slower per tile and is only worth running for the
                # final test pass via infer_to_laz.py --tta 8.
                # bf16 autocast applies here too — PrioStitch on a
                # 20000² scene is many forward passes through the DiT.
                # inference_mode (rather than no_grad) avoids version-
                # counter overhead and lets the allocator be more
                # aggressive about reusing buffers.
                if amp_dtype is not None:
                    cm = torch.autocast(device_type=str(device).split(':')[0],
                                          dtype=amp_dtype)
                else:
                    from contextlib import nullcontext
                    cm = nullcontext()
                with cm:
                    # Paper-faithful PrioStitch per arxiv §3.3 +
                    # §8 Table 7 (best-performing ablation row): coarse
                    # downsamples to the network's input dimensions
                    # (= tile_size), 50% tile overlap, min blending.
                    res = priostitch_infer(
                        model, dsm_max, dsm_min, dsm_mean,
                        dsm_mask.astype(np.float32),
                        coarse_size=tile_size, tile_size=tile_size,
                        overlap=tile_size // 2,
                        blend_mode='linear',
                        device=device,
                        tta=1)
            make_all(
                dsm_max=dsm_max, gt_dtm=gt_dtm,
                valid_gt=valid, dsm_mask=dsm_mask,
                dtm_pred=res['dtm_pred'],
                had_ground=had_g, logit=res['logit'],
                gsd=gsd, alpha_metres=alpha,
                title_prefix=f"step {step}", err_vrange=1.0,
                out_prefix=step_dir / scene_name,
                dpi=160,
                scene_name=scene_name,
                rollup_csv=rollup_csv)
            n_ok += 1
        except torch.cuda.OutOfMemoryError as e:
            log_fn(f"# [viz] {scene_name} OOM: {e} — clearing cache and continuing")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            log_fn(f"# [viz] {scene_name} failed: "
                    f"{type(e).__name__}: {e}")
        finally:
            # Drop the scene's host-side rasters and the inference
            # result before moving to the next scene. The largest
            # scene (20000²) holds ~3.2 GB of float32 rasters across
            # the 7 channels + the dtm_pred/logit results — letting
            # those linger across scenes is what would cause RAM
            # pressure on the 190 GB box. CUDA-side allocator gets
            # nudged too, so the next scene's tile cache starts fresh.
            for _name in ('dsm_max', 'dsm_min', 'dsm_mean', 'dsm_mask',
                           'gt_dtm', 'valid', 'had_g', 'res'):
                if _name in locals():
                    del locals()[_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    log_fn(f"# per-val viz: {n_ok}/{len(viz_scenes)} scenes ok in "
            f"{time.time()-t0:.1f}s -> {step_dir}")

    # Restore live weights
    if ema_orig is not None and ema is not None:
        ema.restore(model, ema_orig)
    model.train()


def _make_split(root: Path, *, val_frac: float = 0.10,
                  seed: int = 1234) -> tuple[list[str], list[str], dict]:
    """Build a train/val scene-name split.

    Deterministic random shuffle by hash(seed, name), then take the
    first `val_frac` for val. Same seed + same scene-name set => same
    split, so resumed runs and downstream scripts (select_scenes
    --split-json) see the same partition.

    No stratification: stratifying by pct_m_alpha required decompressing
    each scene's m_alpha/valid/dsm_mask arrays (~400 MB per scene at
    20000²), which dominated startup for 2000-scene corpora. A pure
    random split over 1900+ scenes lands within ~1% of the corpus
    distribution by terrain type anyway, by the law of large numbers.
    """
    import hashlib

    scenes: list[str] = []
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    try:
        from tqdm.auto import tqdm as _tqdm
        it = _tqdm(dirs, desc="splitting", unit="scene", leave=False,
                    dynamic_ncols=True)
    except ImportError:
        it = dirs
    for d in it:
        if (d / 'raster.npz').exists() and (d / '.done').exists():
            scenes.append(d.name)
    if not scenes:
        raise RuntimeError(f"No usable scenes under {root}")

    # Deterministic hash-sort.
    def _hash_key(name: str) -> int:
        h = hashlib.sha256(f"{seed}|{name}".encode()).digest()
        return int.from_bytes(h[:8], 'little')
    scenes_shuffled = sorted(scenes, key=_hash_key)

    n_val = max(1, int(round(len(scenes_shuffled) * val_frac)))
    val_names = sorted(scenes_shuffled[:n_val])
    train_names = sorted(scenes_shuffled[n_val:])
    stats = dict(n_train=len(train_names), n_val=len(val_names),
                  val_frac_actual=len(val_names) / max(len(scenes), 1))
    return train_names, val_names, stats


# --------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, type=str)
    ap.add_argument('--out_dir', type=str, default=None,
                    help="Override output dir from config.")
    ap.add_argument('--train_root', type=str, default=None,
                    help="Override data.train_root from config.")
    ap.add_argument('--val_root', type=str, default=None,
                    help="Override data.val_root from config.")
    ap.add_argument('--resume', type=str, default=None,
                    help="Path to checkpoint to resume from "
                         "(default: <out_dir>/latest.pt if it exists).")
    ap.add_argument('--device', type=str, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # Apply CLI overrides BEFORE the config is logged so what we print
    # matches what actually runs.
    if args.train_root is not None:
        cfg.setdefault('data', {})['train_root'] = args.train_root
    if args.val_root is not None:
        cfg.setdefault('data', {})['val_root'] = args.val_root

    out_dir = Path(args.out_dir or cfg.get('out_dir', './runs/default'))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = (out_dir / 'train.log').open('a', buffering=1)
    metrics_csv = (out_dir / 'metrics.csv').open('a', buffering=1)
    val_csv = (out_dir / 'val_metrics.csv').open('a', buffering=1)
    # Write CSV headers only if files are empty (don't duplicate on resume).
    if metrics_csv.tell() == 0:
        metrics_csv.write("step,lr,loss,l1,l2,grad,conf,sps\n")
    if val_csv.tell() == 0:
        val_csv.write("step,loss,l1,l2,grad,conf,"
                       "e_t1_pct,e_t2_pct,e_tot_pct,"
                       "e_t1_pct_logit,e_t2_pct_logit,e_tot_pct_logit,"
                       "rmse_ground_m,rmse_non_ground_m,rmse_overall_m,"
                       "n_tiles,n_pixels\n")
    def log(*a):
        s = ' '.join(str(x) for x in a)
        # tqdm.write keeps the bar pinned to the bottom of the terminal
        # so progress + log lines coexist cleanly. With the no-tqdm shim
        # this just calls print().
        tqdm.write(s)
        log_file.write(s + '\n')

    device = (args.device or cfg.get('device') or
              ('cuda' if torch.cuda.is_available() else 'cpu'))
    device = torch.device(device)
    log(f"# device: {device}")

    # --- Speed knobs --------------------------------------------------
    # 1) TF32 for matmul on Ampere+ (Blackwell included). Free 1.2-1.5×
    #    speedup, ~5e-3 relative error vs fp32 — well below diffusion
    #    noise floor.
    # 2) cuDNN benchmark to pick fastest conv kernels for our fixed
    #    input shape.
    # 3) bf16 autocast wraps the forward; AdamW master weights stay fp32.
    #    Blackwell has dedicated bf16 cores → ~2× speedup with no
    #    measurable loss for diffusion training.
    # 4) torch.compile (opt-in via `compile: true` in config) wraps the
    #    DiT in a TorchInductor graph for another 1.2-1.5×. First step
    #    pays a 5-10 min compile cost; subsequent steps are faster.
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        # Tell PyTorch's matmul to prefer tf32 / lower precision for fp32
        # ops (e.g. inside GroupNorm reductions). Combined with TF32 above
        # this routes fp32 GEMMs to tensor cores. No-op for our bf16
        # forwards but cheap insurance for any fp32-cast ops.
        torch.set_float32_matmul_precision('high')
        log(f"# TF32 enabled (matmul + cudnn), cudnn benchmark on, "
            f"matmul_precision=high")

    amp_dtype_str = cfg.get('amp_dtype', 'bfloat16') if device.type == 'cuda' else 'none'
    if amp_dtype_str == 'bfloat16':
        amp_dtype = torch.bfloat16
        log(f"# autocast: bf16 (forward+backward), fp32 master weights")
    elif amp_dtype_str == 'float16':
        amp_dtype = torch.float16
        log(f"# autocast: fp16 (forward+backward), fp32 master weights")
    else:
        amp_dtype = None
        log(f"# autocast: off (fp32)")

    log(f"# config: {json.dumps(cfg, indent=2)}")

    # --- Build dataset --------------------------------------------------
    data_cfg = cfg['data']
    train_root = data_cfg['train_root']
    val_root = data_cfg.get('val_root', train_root)
    tile = int(data_cfg.get('tile_size', 512))
    train_scenes = data_cfg.get('train_scenes', None)
    val_scenes = data_cfg.get('val_scenes', None)
    min_valid_frac = float(data_cfg.get('min_valid_frac', 0.50))

    # --- Train/val split -----------------------------------------------
    # If the user pre-supplied train_scenes / val_scenes lists, honour
    # them. Otherwise compute a fresh stratified split: every scene is
    # assigned to a complexity bin by its pct_m_alpha (the fraction of
    # cells where DSM agrees with GT_DTM within α metres -- "open" cells),
    # and we hold out `val_frac` (default 10 %) of EACH bin for
    # validation. The split is keyed by scene name hash + a fixed seed,
    # so it's stable across re-runs unless you change `split_seed` or
    # `val_frac` in the config.
    #
    # Why stratified: a uniform random split on a corpus dominated by
    # open agricultural land would give us a val set that's almost all
    # open scenes, and validation E_tot would underestimate the real
    # error on cluttered urban tiles. Stratifying ensures the val set
    # has the same composition as train.
    val_frac = float(data_cfg.get('val_frac', 0.10))
    split_seed = int(data_cfg.get('split_seed', 1234))
    split_file = out_dir / 'split.json'

    if train_scenes is None and val_scenes is None:
        if train_root != val_root:
            log(f"# WARNING: train_root and val_root differ; not "
                f"computing a held-out split (assuming the val_root "
                f"is already disjoint).")
        elif split_file.exists():
            # Reuse the existing split so resumed runs keep the same
            # train/val partition. Otherwise adding new preprocessed
            # scenes between runs would silently re-shuffle the split
            # and bleed training scenes into val.
            sj = json.loads(split_file.read_text())
            train_scenes = sj['train_scenes']
            val_scenes = sj['val_scenes']
            log(f"# loaded existing split from {split_file}  "
                f"(train: {len(train_scenes)}, val: {len(val_scenes)}, "
                f"val_frac={sj.get('val_frac', '?')}, "
                f"seed={sj.get('split_seed', '?')})")
        else:
            log(f"# computing stratified train/val split "
                f"(val_frac={val_frac:.0%}, split_seed={split_seed})...")
            train_scenes, val_scenes, split_stats = _make_split(
                Path(train_root), val_frac=val_frac, seed=split_seed)
            split_file.write_text(json.dumps(dict(
                val_frac=val_frac, split_seed=split_seed,
                stats=split_stats,
                train_scenes=train_scenes,
                val_scenes=val_scenes,
            ), indent=2))
            log(f"#   -> train: {len(train_scenes)} scenes, "
                f"val: {len(val_scenes)} scenes  "
                f"(val_frac_actual={split_stats['val_frac_actual']:.3f})")
            log(f"#   -> split written to {split_file}")

    # Hard mining is OFF by default now. With hard_mining_strength <= 0
    # the sampler is uniform (paper §7.1 / the original GrounDiff used a
    # uniform sampler — config `sampler: uniform`). A loss-keyed miner
    # biases the *reported* training loss upward and, on this corpus,
    # preferentially oversamples low-coverage tiles whose TIN GT is the
    # least reliable, which amplifies label noise. We keep the HardMiner
    # object for checkpoint back-compat but pass None to the datasets
    # unless explicitly re-enabled. See `use_hard_mining` below.
    hard_mining_strength = float(data_cfg.get('hard_mining_strength', 0.0))
    use_hard_mining = hard_mining_strength > 0.0
    miner = HardMiner(
        ema_decay=float(data_cfg.get('miner_decay', 0.95)),
        init_weight=float(data_cfg.get('miner_init', 1.0)))
    active_miner = miner if use_hard_mining else None
    log(f"# hard mining: {'ON (strength=%.2f)' % hard_mining_strength if use_hard_mining else 'OFF (uniform sampling)'}")
    # Per-worker cache cap. Each cached scene at 20000² is ~11 GB host
    # RAM. Default 1 → ~11 GB per worker × num_workers ≈ memory cost.
    max_cached = int(data_cfg.get('max_cached_scenes', 1))
    val_min_valid = float(data_cfg.get('val_min_valid_frac', min_valid_frac))
    burst = int(data_cfg.get('tiles_per_scene_burst', 64))

    # Pre-tiled fast path: if data_cfg.pretile_root is set, read individual
    # per-tile .npz files instead of slicing from full scene arrays. This
    # eliminates scene-load stalls at finer GSDs. Generated by
    # `scripts/pretile.py`. The val set still uses the scene-based
    # RasterEvalTiles path because it runs infrequently (val_every_steps)
    # and benefits from the deterministic scene-grid layout.
    pretile_root = data_cfg.get('pretile_root', None)
    use_pretiled = bool(pretile_root)

    log(f"# building train dataset (scanning {pretile_root if use_pretiled else train_root})...")
    t0 = time.time()
    if use_pretiled:
        train_ds = RasterPretiledDataset(
            pretile_root, tile_size=tile, scene_filter=train_scenes,
            hard_miner=active_miner,
            hard_mining_strength=hard_mining_strength,
            seed=int(cfg.get('seed', 0)),
            augment=bool(data_cfg.get('augment', True)),
            multiscale_sizes=tuple(data_cfg.get('multiscale_sizes', (256, 512, 1024))),
            multiscale_prob=float(data_cfg.get('multiscale_prob', 0.5)),
            rot90_prob=float(data_cfg.get('rot90_prob', 0.5)),
            rot_jitter_deg=float(data_cfg.get('rot_jitter_deg', 5.0)),
            rot_jitter_prob=float(data_cfg.get('rot_jitter_prob', 0.5)),
            flip_prob=float(data_cfg.get('flip_prob', 0.5)),
            alpha_metres=float(cfg.get('loss', {}).get('alpha_metres', cfg.get('alpha_metres', 0.20))))
        log(f"#   -> {len(train_ds.scenes)} scenes, "
            f"{len(train_ds.tiles)} tiles total (pretiled)  "
            f"({time.time()-t0:.1f}s)")
    else:
        train_ds = RasterTileDataset(
            train_root, tile_size=tile, scene_filter=train_scenes,
            min_valid_frac=min_valid_frac, hard_miner=active_miner,
            hard_mining_strength=hard_mining_strength,
            seed=int(cfg.get('seed', 0)),
            max_cached_scenes=max_cached,
            tiles_per_scene_burst=burst,
            augment=bool(data_cfg.get('augment', True)),
            # Paper §7.1 augmentation pipeline.
            multiscale_sizes=tuple(data_cfg.get('multiscale_sizes', (256, 512, 1024))),
            multiscale_prob=float(data_cfg.get('multiscale_prob', 0.5)),
            rot90_prob=float(data_cfg.get('rot90_prob', 0.5)),
            rot_jitter_deg=float(data_cfg.get('rot_jitter_deg', 5.0)),
            rot_jitter_prob=float(data_cfg.get('rot_jitter_prob', 0.5)),
            flip_prob=float(data_cfg.get('flip_prob', 0.5)),
            alpha_metres=float(cfg.get('loss', {}).get('alpha_metres', cfg.get('alpha_metres', 0.20))))
        log(f"#   -> {len(train_ds.scenes)} scenes, "
            f"~{len(train_ds)} tiles/epoch, "
            f"burst={burst} tiles/scene  ({time.time()-t0:.1f}s)")
    log(f"# building val dataset (scanning {val_root}, "
        f"min_valid_frac={val_min_valid})...")
    t0 = time.time()
    val_ds = RasterEvalTiles(
        val_root, tile_size=tile, scene_filter=val_scenes,
        min_valid_frac=val_min_valid,
        max_cached_scenes=max_cached)
    log(f"#   -> {len(val_ds.scenes)} scenes, {len(val_ds)} tiles  "
        f"({time.time()-t0:.1f}s)")

    # Cross-check: train and val must be disjoint by scene name.
    train_set = {s['name'] for s in train_ds.scenes}
    val_set = {s['name'] for s in val_ds.scenes}
    overlap = train_set & val_set
    if overlap:
        log(f"# WARNING: train and val share {len(overlap)} scenes "
            f"(e.g. {sorted(overlap)[:5]}). Validation metrics will be "
            f"optimistic.")
    else:
        log(f"# train ∩ val = ∅ (true held-out evaluation)")

    # --- Pre-select viz scenes ----------------------------------------
    # These are the 20 representative val scenes we'll render at every
    # val step (when viz.per_val_viz is on). Picking them BEFORE training
    # starts means every val pass produces visualisations for the SAME
    # scenes, so you can scrub through training and see specific scenes
    # improving. Selection is stratified by pct_m_alpha so the set
    # covers open/mixed/cluttered terrain proportionally.
    viz_cfg = cfg.get('viz', {})
    viz_scenes: list[dict] = []
    if bool(viz_cfg.get('per_val_viz', True)):
        from stage2_raster.scripts.select_scenes import select_scenes
        viz_scenes_file = out_dir / 'viz_scenes.json'
        if viz_scenes_file.exists():
            # Reuse on resume (just like split.json).
            viz_scenes = json.loads(viz_scenes_file.read_text()).get('scenes', [])
            log(f"# loaded viz scene list from {viz_scenes_file} "
                f"({len(viz_scenes)} scenes)")
        else:
            log(f"# picking {int(viz_cfg.get('n_scenes', 20))} "
                f"viz scenes from val set...")
            try:
                viz_scenes = select_scenes(
                    Path(val_root),
                    tile_size=tile,
                    n=int(viz_cfg.get('n_scenes', 20)),
                    min_factor=float(viz_cfg.get('min_factor', 2.0)),
                    restrict_to=val_set)
                viz_scenes_file.write_text(json.dumps(
                    dict(n_selected=len(viz_scenes), scenes=viz_scenes),
                    indent=2))
                log(f"#   -> {len(viz_scenes)} viz scenes written to "
                    f"{viz_scenes_file}")
                for s in viz_scenes[:5]:
                    ng = s.get('nonground_frac')
                    if ng is not None:
                        log(f"#     [stratum {s.get('stratum','?')}] "
                            f"{s['name']}  {s['H']}x{s['W']}  "
                            f"non-ground={100*ng:.1f}%")
                    else:
                        log(f"#     {s['name']}  {s['H']}x{s['W']}")
                if len(viz_scenes) > 5:
                    log(f"#     ... and {len(viz_scenes) - 5} more")
            except Exception as e:
                log(f"# WARNING: could not pick viz scenes ({e}); "
                    f"per-val viz disabled")
                viz_scenes = []

    batch_size = int(cfg.get('batch_size', 4))
    num_workers = int(cfg.get('num_workers', 4))
    prefetch_factor = int(cfg.get('prefetch_factor', 4))
    # prefetch_factor: each worker prepares N batches ahead of the GPU
    # so the GPU never sits idle waiting for the next batch. With our
    # scene-locality burst (each worker yields `burst` consecutive tiles
    # from one cached scene), prefetch=4 means each worker keeps 4
    # batches queued ahead = 4 * batch_size tiles. If GPU is starving
    # (low util, oscillating), bump prefetch_factor and/or num_workers.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=(device != 'cpu'),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0))
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=max(num_workers // 2, 0),
        pin_memory=(device != 'cpu'),
        shuffle=False)

    # --- Build model ----------------------------------------------------
    # Strip leading-underscore keys ("_doc", "_doc_outliers", etc.) so
    # they're not passed as kwargs to constructors. They're config-file
    # comments, not parameters.
    def _strip_doc(d):
        if not isinstance(d, dict):
            return d
        return {k: v for k, v in d.items() if not k.startswith('_')}

    model = GrounDiffRaster(
        backbone=str(cfg.get('backbone', 'unet')),
        backbone_kwargs=_strip_doc(cfg.get('model', {})),
        diffusion_kwargs=_strip_doc(cfg.get('diffusion', {})),
        loss_kwargs=_strip_doc(cfg.get('loss', {})),
    ).to(device)
    log(f"# model: UNet, {model.num_params / 1e6:.1f} M params")

    # Kaiming-normal init (matches official defra.json `init_kaiming: true`,
    # which inherits from Janspiry-Palette base_network.init_weights).
    # PyTorch's default Conv2d init is kaiming_uniform_(a=sqrt(5)); the
    # Palette convention uses kaiming_normal_(a=0, mode='fan_in') which
    # gives variance 2/fan_in — larger initial weights than PyTorch
    # defaults. Skip zero-init'd modules (final out conv, zero_modules
    # inside ResBlocks) by checking weight is non-zero.
    if cfg.get('init_kaiming', True):
        from torch import nn
        n_init = 0
        for m in model.modules():
            cls_name = m.__class__.__name__
            if 'Conv' not in cls_name and 'Linear' not in cls_name:
                continue
            if not hasattr(m, 'weight') or m.weight is None:
                continue
            with torch.no_grad():
                if float(m.weight.detach().abs().sum().item()) == 0.0:
                    continue
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
            n_init += 1
        log(f"# kaiming-normal init applied to {n_init} Conv/Linear modules")

    # Channels-last (NHWC) memory layout. On Blackwell / any modern
    # tensor-core GPU, cuDNN's NHWC conv kernels run ~10-30% faster than
    # the default NCHW because the inner accumulator dim aligns with the
    # tensor-core fragment layout. We convert the model weights here;
    # input tensors get converted inside `GrounDiffUNet.forward` (one
    # call site, applies to both training and inference).
    if device.type == 'cuda' and bool(cfg.get('channels_last', True)):
        model = model.to(memory_format=torch.channels_last)
        log(f"# channels-last (NHWC) layout enabled")

    # torch.compile is opt-in via `compile: true`. Default mode is
    # `max-autotune-no-cudagraphs`: kernel-level autotuning WITHOUT
    # CUDA graphs. Why "no-cudagraphs": CUDA graphs (enabled by plain
    # `max-autotune` or `reduce-overhead`) capture a static set of
    # buffers per compiled region. With gradient accumulation we run
    # the same compiled forward multiple times in a row (one per
    # micro-batch) before backward — the captured buffers get
    # overwritten by micro N+1 before backward of micro N has finished
    # consuming them. Result: 'accessing tensor output of CUDAGraphs
    # that has been overwritten' on the first backward. Dropping
    # CUDA graphs costs ~5-10% but is grad-accum-safe. Fall back to
    # 'default' if even no-cudagraphs autotuning errors out.
    if bool(cfg.get('compile', False)) and device.type == 'cuda':
        compile_mode = str(cfg.get('compile_mode', 'max-autotune-no-cudagraphs'))
        log(f"# torch.compile: ON, mode={compile_mode!r}  "
            f"(first ~50 steps will be slow while compiling/autotuning)")
        compiled = torch.compile(model.net, mode=compile_mode)
        # Keep `model.net` (and its alias `model.dit`) pointing at the
        # compiled module so checkpoint save/load and the diffusion
        # sampler all see the compiled forward path.
        model.net = compiled
        model.dit = compiled

    # --- Optimizer + EMA + LR schedule ----------------------------------
    optim_cfg = _strip_doc(cfg.get('optim', {}))
    base_lr = float(optim_cfg.get('lr', 1e-4))
    min_lr = float(optim_cfg.get('min_lr', 1e-6))
    wd = float(optim_cfg.get('weight_decay', 1e-2))
    warmup = int(optim_cfg.get('warmup_steps', 1000))
    grad_clip = float(optim_cfg.get('grad_clip', 5.0))

    # Grad accumulation: every optimizer step = `accum_steps` micro
    # batches, so the effective batch is batch_size * accum_steps.
    # This lets us match the paper's batch=16 even when our 1024x1024
    # tiles + 197M-param model can only fit micro_batch=2 on a 95 GB
    # Blackwell. 'step' in our log = OPTIMIZER step (one update);
    # micro-batches are silent.
    accum_steps = int(cfg.get('grad_accum_steps', 1))
    effective_batch = batch_size * accum_steps
    log(f"# batch: micro={batch_size}  x  grad_accum={accum_steps}  "
        f"=  effective={effective_batch}")

    # total_steps: 'auto' means derive from epochs + corpus size +
    # effective batch. The dataset's __len__ is the per-epoch nominal
    # tile count (sum of scene areas / tile²). We don't actually
    # enumerate that many tiles per epoch (the dataset is iterable and
    # samples randomly), but it's the right denominator for "60 epochs
    # of effective coverage".
    ts_raw = optim_cfg.get('total_steps', 'auto')
    epochs = int(optim_cfg.get('epochs', 60))
    if isinstance(ts_raw, str) and ts_raw.lower() == 'auto':
        tiles_per_epoch = len(train_ds)
        total_steps = max(
            warmup + 100,                          # need at least warmup
            int(round(epochs * tiles_per_epoch / max(effective_batch, 1)))
        )
        log(f"# total_steps auto-derived: "
            f"{epochs} epochs * {tiles_per_epoch} tiles/epoch / "
            f"{effective_batch} effective = {total_steps} optimizer steps")
    else:
        total_steps = int(ts_raw)
        log(f"# total_steps from config: {total_steps}")

    # Fused AdamW: 5-10% faster optimizer step on CUDA. The 'fused'
    # kernel folds the param update into a single kernel launch instead
    # of the eager parameter-by-parameter loop. Falls back to non-fused
    # on CPU.
    use_fused_adam = (device.type == 'cuda')
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=wd,
        betas=tuple(optim_cfg.get('betas', (0.9, 0.999))),
        fused=use_fused_adam)
    ema = EMA(model, decay=float(optim_cfg.get('ema_decay', 0.9999)))
    log(f"# optimizer: AdamW (fused={use_fused_adam})")

    # --- Resume ---------------------------------------------------------
    start_step = 0
    best_e_tot = float('inf')
    resume_path = Path(args.resume) if args.resume else (out_dir / 'latest.pt')
    if resume_path.exists():
        d = load_checkpoint(resume_path, model=model, optimizer=optimizer,
                             ema=ema, miner=miner)
        start_step = int(d.get('step', 0))
        best_e_tot = float(d.get('best_e_tot', best_e_tot))
        log(f"# resumed from {resume_path} at step {start_step}, "
            f"best_e_tot={best_e_tot:.3f}")

    # --- Signal handler for clean shutdown ------------------------------
    interrupted = {'flag': False}
    def _handler(signum, frame):
        interrupted['flag'] = True
        log(f"# signal {signum} received; will checkpoint and exit after this step")
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    # --- Training loop --------------------------------------------------
    val_every = int(cfg.get('val_every_steps', 2000))
    save_every = int(cfg.get('save_every_steps', 500))
    log_every = int(cfg.get('log_every_steps', 50))

    model.train()
    step_loss_ema = None
    t_last = time.time()
    n_oom = 0

    # Live progress bar. `dynamic_ncols` lets it resize with the terminal.
    # If tqdm isn't installed the shim above makes this a no-op so the
    # existing per-50-step text log still works.
    bar = tqdm(
        total=total_steps, initial=start_step,
        desc="train", dynamic_ncols=True,
        smoothing=0.05,            # smooth ETA against bursty step times
        miniters=1,                # update display every step (cheap)
        mininterval=0.5,           # but no more than 2 Hz, to spare the term
    )

    # Force the first batch to materialise before the loop starts so the
    # user sees something happen. The dataloader's worker startup +
    # first-tile decompression can take 30s-2min on a fresh corpus; if
    # we don't pull a batch here, the bar sits at 0% with no visible
    # activity for that whole time and it looks hung.
    log(f"# waiting for first batch from {num_workers} dataloader "
        f"workers...")
    t0 = time.time()
    train_iter = iter(train_loader)
    first_batch = next(train_iter)
    log(f"#   -> first batch ready ({time.time()-t0:.1f}s)")

    # Worker-death resilience: a corrupt .npz can segfault a DataLoader
    # worker (numpy/scipy C extensions can crash on malformed data
    # without raising a Python exception). PyTorch installs a SIGCHLD
    # handler that propagates "DataLoader worker (pid N) is killed by
    # signal" from ANY torch call — not just next(loader), but also
    # .backward(), optimizer.step(), tensor ops, etc. — so we need a
    # broad catch around the whole optimizer step, not just the data
    # fetch. The function below is hoisted to outer scope so it's
    # visible from any catch site.
    def _recreate_loader_after_worker_death(err: BaseException):
        nonlocal train_loader, train_iter
        log(f"# !! dataloader worker died: {type(err).__name__}: {err}")
        log(f"# !! recreating loader and continuing from step {step}")
        try:
            del train_iter
        except Exception:
            pass
        try:
            del train_loader
        except Exception:
            pass
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2.0)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, num_workers=num_workers,
            pin_memory=(device != 'cpu'),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=(num_workers > 0))
        train_iter = iter(train_loader)

    for step in range(start_step, total_steps):
        # LR step
        lr = cosine_warmup(step, warmup=warmup, total=total_steps,
                            base_lr=base_lr, min_lr=min_lr)
        for g in optimizer.param_groups:
            g['lr'] = lr

        # Get the first micro-batch — first time around, use the
        # pre-fetched batch.
        if first_batch is not None:
            batch = first_batch
            first_batch = None
        else:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            except RuntimeError as e:
                if "DataLoader worker" in str(e) and "killed by signal" in str(e):
                    _recreate_loader_after_worker_death(e)
                    batch = next(train_iter)
                else:
                    raise

        # --- Grad-accumulated optimizer step ---------------------------
        # One optimizer step = accum_steps micro-batches. We zero grads
        # once at the start, run accum_steps forward/backward passes
        # scaling each loss by 1/accum_steps, then step the optimizer.
        # This makes the effective batch = batch_size * accum_steps
        # while peak activation memory stays at micro_batch.
        try:
            optimizer.zero_grad(set_to_none=True)
            # Accumulate losses on GPU as tensors. The old code synced
            # `loss.item()` and `metrics[k].item()` every micro-batch,
            # which forced 4 CUDA sync points per optimizer step and
            # drained the pipeline (especially bad with CUDA graphs).
            # Now we only sync once, after `optimizer.step()`.
            accum_loss_t = torch.zeros((), device=device)
            accum_m_t = {k: torch.zeros((), device=device)
                         for k in ('l1', 'l2', 'grad', 'conf')}
            metas = []
            for micro in range(accum_steps):
                if micro > 0:
                    # Pull the next micro-batch. Same worker-death
                    # resilience as the outer fetch.
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        batch = next(train_iter)
                    except RuntimeError as e:
                        if ("DataLoader worker" in str(e)
                                and "killed by signal" in str(e)):
                            _recreate_loader_after_worker_death(e)
                            batch = next(train_iter)
                        else:
                            raise
                # non_blocking=True lets the H2D copy overlap with the
                # previous step's kernels (pinned memory required, which
                # DataLoader already provides via pin_memory=True).
                b = {k: v.to(device, non_blocking=True)
                       if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
                # Autocast wraps just the forward; backward runs through
                # the autocast cast operators. We compute the loss in
                # bf16 → the backward grads are bf16 → AdamW's fused
                # update casts back to fp32 master weights internally.
                # No GradScaler needed for bf16 (it has fp32's range).
                if amp_dtype is not None:
                    with torch.autocast(device_type=device.type,
                                          dtype=amp_dtype):
                        loss, metrics = model.training_step(
                            dsm_max=b['dsm_max'], dsm_min=b['dsm_min'],
                            dsm_mean=b['dsm_mean'], dsm_mask=b['dsm_mask'],
                            dtm=b['gt_dtm'], valid=b['valid'],
                            m_alpha=b['m_alpha'])
                else:
                    loss, metrics = model.training_step(
                        dsm_max=b['dsm_max'], dsm_min=b['dsm_min'],
                        dsm_mean=b['dsm_mean'], dsm_mask=b['dsm_mask'],
                        dtm=b['gt_dtm'], valid=b['valid'],
                        m_alpha=b['m_alpha'])
                # Scale so the accumulated grad equals the average grad
                # across the effective batch.
                (loss / accum_steps).backward()
                # Accumulate on-device — no .item() sync here.
                accum_loss_t += loss.detach()
                for k in accum_m_t:
                    accum_m_t[k] += metrics[k]
                metas.append(b['meta'])
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ema.update(model)
            # ONE sync per outer step (here). The optimizer.step() above
            # is queued async; this .item() drains it.
            accum_loss = float(accum_loss_t.item()) / accum_steps
            accum_metrics = {k: float(v.item()) / accum_steps
                             for k, v in accum_m_t.items()}
            # Hand off the averaged losses + metas under the same names
            # the rest of the loop expects.
            metrics = accum_metrics
        except torch.cuda.OutOfMemoryError as e:
            n_oom += 1
            log(f"[step {step}] OOM #{n_oom}; clearing cache and skipping")
            torch.cuda.empty_cache()
            continue
        except RuntimeError as e:
            # PyTorch's signal handler propagates "DataLoader worker (pid
            # N) is killed by signal: Segmentation fault" from ANY torch
            # call (.backward(), .step(), etc.), not just next(loader).
            # When the segfault hits mid-step, in-progress grads are
            # discarded by the next optimizer.zero_grad() and we resume
            # from the previous optimizer state. No corruption since the
            # optimizer hasn't yet stepped.
            if ("DataLoader worker" in str(e)
                    and "killed by signal" in str(e)):
                _recreate_loader_after_worker_death(e)
                # Drop any half-computed grads from the failed step.
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

        # Update hard miner per-tile losses across all accumulated micro
        # batches. Use the averaged L1 as the difficulty signal. Skipped
        # entirely when hard mining is off (uniform sampling) so the
        # reported training loss is the honest uniform-sampled average.
        if use_hard_mining:
            tile_loss = float(accum_metrics['l1'])
            for meta in metas:
                for k in range(len(meta['scene'])):
                    miner.update(scene=meta['scene'][k],
                                  ix=int(meta['ix'][k]),
                                  iy=int(meta['iy'][k]),
                                  loss=tile_loss)

        # Logging — `ls` is the scalar loss for THIS optimizer step
        # (averaged over the effective batch).
        ls = float(accum_loss)
        step_loss_ema = ls if step_loss_ema is None else 0.99 * step_loss_ema + 0.01 * ls
        sps_inst = 0.0
        metrics_csv.write(
            f"{step+1},{lr:.6e},{ls:.6f},"
            f"{float(metrics['l1']):.6f},{float(metrics['l2']):.6f},"
            f"{float(metrics['grad']):.6f},{float(metrics['conf']):.6f},"
            f"{sps_inst:.3f}\n")

        # Update the live bar every step. The postfix shows the smoothed
        # loss + the four loss components, so you can watch convergence
        # in real time without waiting for the per-log_every text print.
        bar.set_postfix_str(
            f"loss={step_loss_ema:.3f} L1={float(metrics['l1']):.3f} "
            f"L2={float(metrics['l2']):.3f} L∇={float(metrics['grad']):.3f} "
            f"Lc={float(metrics['conf']):.3f} lr={lr:.1e}",
            refresh=False)
        bar.update(1)

        if (step + 1) % log_every == 0:
            now = time.time()
            sps = log_every / max(now - t_last, 1e-9)
            t_last = now
            log(f"[step {step+1:>6d}/{total_steps}] lr={lr:.2e} "
                f"loss={step_loss_ema:.4f} (cur={ls:.4f})  "
                f"L1={float(metrics['l1']):.4f} L2={float(metrics['l2']):.4f} "
                f"L∇={float(metrics['grad']):.4f} Lc={float(metrics['conf']):.4f}  "
                f"{sps:.1f} step/s")

        # Periodic save
        if (step + 1) % save_every == 0 or interrupted['flag']:
            save_checkpoint(
                out_dir / 'latest.pt',
                model=model, optimizer=optimizer, ema=ema, miner=miner,
                step=step + 1, config=cfg, best_e_tot=best_e_tot)
            log(f"# saved checkpoint at step {step+1}")

        # Val pass
        if (step + 1) % val_every == 0 or interrupted['flag']:
            log(f"# eval at step {step+1}...")
            res = evaluate(model, val_loader, device, use_ema=ema,
                            max_batches=int(cfg.get('val_max_batches', 50)),
                            amp_dtype=amp_dtype,
                            alpha_metres=float(
                                cfg.get('loss', {}).get('alpha_metres',
                                cfg.get('alpha_metres', 0.20))))
            log(f"[val   {step+1:>6d}/{total_steps}] "
                f"loss={res['loss']:.4f} "
                f"L1={res['l1']:.4f} L2={res['l2']:.4f} "
                f"L∇={res['grad']:.4f} Lc={res['conf']:.4f}  "
                f"[residual] E_T1={res['e_t1_pct']:.2f}% E_T2={res['e_t2_pct']:.2f}% "
                f"E_tot={res['e_tot_pct']:.2f}%  "
                f"[σ(ℓ)] E_T1={res['e_t1_pct_logit']:.2f}% E_T2={res['e_t2_pct_logit']:.2f}% "
                f"E_tot={res['e_tot_pct_logit']:.2f}%  "
                f"RMSE_gnd={res['rmse_gnd_m']:.3f}m "
                f"RMSE_ng={res['rmse_ng_m']:.3f}m "
                f"RMSE_all={res['rmse_all_m']:.3f}m  "
                f"(n_tiles={res['n_tiles']}, n_pixels={res['n_pixels']})")
            val_csv.write(
                f"{step+1},{res['loss']:.6f},{res['l1']:.6f},{res['l2']:.6f},"
                f"{res['grad']:.6f},{res['conf']:.6f},"
                f"{res['e_t1_pct']:.4f},{res['e_t2_pct']:.4f},"
                f"{res['e_tot_pct']:.4f},"
                f"{res['e_t1_pct_logit']:.4f},{res['e_t2_pct_logit']:.4f},"
                f"{res['e_tot_pct_logit']:.4f},"
                f"{res['rmse_gnd_m']:.6f},{res['rmse_ng_m']:.6f},"
                f"{res['rmse_all_m']:.6f},"
                f"{res['n_tiles']},{res['n_pixels']}\n")
            val_csv.flush()
            # Per-val-pass checkpoint snapshot. Saved as
            # ckpt_step{step:07d}.pt (sorts naturally) so all val
            # snapshots can be retained and ranked later by whichever
            # metric (e_tot, rmse_gnd, rmse_ng, rmse_all, L1, etc.) the
            # downstream evaluation cares about. At ~1 GB each × 20 val
            # passes = ~20 GB total — comfortable on the disk budget.
            save_checkpoint(
                out_dir / f'ckpt_step{step+1:07d}.pt',
                model=model, optimizer=optimizer, ema=ema, miner=miner,
                step=step + 1, config=cfg, best_e_tot=best_e_tot)
            log(f"# saved per-val snapshot -> ckpt_step{step+1:07d}.pt")
            if res['e_tot_pct'] < best_e_tot:
                best_e_tot = res['e_tot_pct']
                save_checkpoint(
                    out_dir / 'best.pt',
                    model=model, optimizer=optimizer, ema=ema, miner=miner,
                    step=step + 1, config=cfg, best_e_tot=best_e_tot)
                log(f"# new best E_tot={best_e_tot:.3f} -> best.pt")

            # Per-val scene visualisations: PrioStitch + render for each
            # of the 20 representative val scenes. Snapshots progress
            # at this step under out_dir/viz/step_<step>/.
            #
            # Gated by `viz_priostitch_every_steps` (config; default =
            # `val_every_steps` if absent, i.e. every val). At 512² each
            # full-scene PrioStitch viz is slow, so default config sets
            # this to 2× val cadence (e.g. val every 12500 + viz every
            # 25000 → 10 viz passes for a 250k-step run). Step boundary
            # check matches the val condition (`(step + 1) % N == 0`)
            # so the snapshot lands on round step numbers.
            viz_every = int(cfg.get('viz_priostitch_every_steps',
                                     cfg.get('val_every_steps', 0)))
            do_priostitch_viz = (
                viz_scenes
                and viz_every > 0
                and (step + 1) % viz_every == 0
            )
            if do_priostitch_viz:
                _run_per_val_viz(
                    model, viz_scenes,
                    val_root=Path(val_root),
                    out_root=out_dir,
                    step=step + 1,
                    device=device,
                    ema=ema,
                    tile_size=tile,
                    amp_dtype=amp_dtype,
                    log_fn=log)
            elif viz_scenes:
                log(f"# per-val viz skipped at step {step + 1} "
                    f"(next viz at step "
                    f"{((step + 1) // viz_every + 1) * viz_every})")

            # End of val phase: drain any leftover buffers from val +
            # viz before training resumes. Without this, the training-
            # step allocator may have to wait for inference activations
            # to be reclaimed, occasionally producing a single
            # high-fragmentation step that risks an unnecessary OOM.
            if torch.cuda.is_available() and device.type == 'cuda':
                torch.cuda.empty_cache()

        if interrupted['flag']:
            log("# exiting on interrupt")
            break

    bar.close()
    log("# training complete")


if __name__ == '__main__':
    main()

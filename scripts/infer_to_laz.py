#!/usr/bin/env python3
"""Apply trained GrounDiffRaster to a LAZ scene, writing back per-point
ground/non-ground classification and the predicted DTM height ẑ(x, y).

Workflow:
  1. Load preprocessed raster.npz (built by data/preprocess.py).
  2. Run PrioStitch inference → dtm_pred (H, W) in metres.
  3. Load original LAZ.
  4. For each ALS point (x, y, z): bilinearly sample dtm_pred to get
     ẑ(x, y), then classify as ground iff `|z − ẑ| < alpha_metres`
     (Sithole-Vosselman 2003 §4.2.1 with the same threshold we built
     M_α with).
  5. Write a new LAZ with:
       - classification = 2 (ground) or 1 (non-ground) per point
       - extra dimensions:
           'dtm_pred_z'    : float32, ẑ at each point
           'residual_z'    : float32, z − ẑ
           'prob_ground'   : uint16, σ(ℓ) * 65535 at each point
           'gt_class'      : uint8, original LAZ classification

Usage:
    python -m stage2_raster.scripts.infer_to_laz \
        --raster /path/to/scene/raster.npz \
        --laz /path/to/scene.laz \
        --ckpt /path/to/best.pt \
        --out /path/to/out.laz \
        --alpha 0.20

Optionally `--coarse-size 1024 --tile-size 1024 --overlap 256`. Defaults
match the production config.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage2_raster.models import GrounDiffRaster
from stage2_raster.utils.priostitch import priostitch_infer


def bilinear_sample(raster: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                     bbox: Tuple[float, float, float, float],
                     gsd: float) -> np.ndarray:
    """Bilinearly sample `raster` (H, W) at world coordinates (xs, ys).

    `raster` is in row-major (H, W) with row 0 at y=ymax (north-up).
    Returns float32 array, same shape as xs/ys.
    """
    xmin, ymin, xmax, ymax = bbox
    H, W = raster.shape

    # World -> pixel coordinate (real-valued).
    # row coord increases southward; row = (ymax - y) / gsd - 0.5
    # col coord = (x - xmin) / gsd - 0.5
    rr = (ymax - ys) / gsd - 0.5
    cc = (xs - xmin) / gsd - 0.5

    # Clamp to valid range.
    r0 = np.clip(np.floor(rr).astype(np.int64), 0, H - 1)
    r1 = np.clip(r0 + 1, 0, H - 1)
    c0 = np.clip(np.floor(cc).astype(np.int64), 0, W - 1)
    c1 = np.clip(c0 + 1, 0, W - 1)
    fr = np.clip(rr - r0, 0.0, 1.0).astype(np.float32)
    fc = np.clip(cc - c0, 0.0, 1.0).astype(np.float32)

    v00 = raster[r0, c0]
    v01 = raster[r0, c1]
    v10 = raster[r1, c0]
    v11 = raster[r1, c1]
    v0 = v00 * (1 - fc) + v01 * fc
    v1 = v10 * (1 - fc) + v11 * fc
    v = v0 * (1 - fr) + v1 * fr
    return v.astype(np.float32)


def _write_laz_with_classification(in_laz: Path, out_laz: Path,
                                    cls_new: np.ndarray,
                                    dtm_pred_z: np.ndarray,
                                    residual_z: np.ndarray,
                                    prob_ground_u16: np.ndarray):
    """Write a plain (non-COPC) LAZ carrying the new classification plus
    three extra dims (predicted DTM z, residual, ground prob), copying the
    core point data from `in_laz`.

    Why we build a FRESH header instead of copying the input object:
    DEFRA national-LiDAR tiles are distributed as COPC (Cloud-Optimised
    Point Cloud) LAZ. laspy can READ COPC but cannot WRITE it back —
    `las.write()` on a COPC-sourced object raises
    'NotImplementedError: Writing COPC is not supported'. We therefore
    create a standard LasData with the same point format / scales / offset
    / CRS and copy the dimensions across, yielding a normal LAZ that any
    reader (incl. QGIS, PDAL, CloudCompare) opens fine."""
    import laspy

    with laspy.open(str(in_laz)) as f:
        src = f.read()

    # Fresh, non-COPC header. Keep the source's point format id, scales,
    # offsets and (if present) CRS/VLRs that aren't COPC-specific.
    src_hdr = src.header
    header = laspy.LasHeader(
        version=src_hdr.version,
        point_format=src_hdr.point_format)
    header.scales = src_hdr.scales
    header.offsets = src_hdr.offsets
    # Preserve CRS where available (laspy >=2.3 exposes parse/add helpers).
    try:
        crs = src_hdr.parse_crs()
        if crs is not None:
            header.add_crs(crs)
    except Exception:
        pass

    # Register the extra dims on the new header.
    extra = [
        laspy.ExtraBytesParams(name='dtm_pred_z', type=np.float32,
                               description='stage2 predicted DTM z (m)'),
        laspy.ExtraBytesParams(name='residual_z', type=np.float32,
                               description='stage2 z - dtm_pred (m)'),
        laspy.ExtraBytesParams(name='prob_ground', type=np.uint16,
                               description='stage2 P(ground) * 65535'),
        laspy.ExtraBytesParams(name='gt_class', type=np.uint8,
                               description='original classification'),
    ]
    try:
        header.add_extra_dims(extra)
    except AttributeError:                      # older laspy: add one-by-one
        for p in extra:
            header.add_extra_dim(p)

    out = laspy.LasData(header)

    # Copy the standard point dimensions present in BOTH source and dest.
    copy_dims = ('X', 'Y', 'Z', 'intensity', 'return_number',
                 'number_of_returns', 'scan_direction_flag',
                 'edge_of_flight_line', 'synthetic', 'key_point',
                 'withheld', 'scan_angle_rank', 'scan_angle',
                 'user_data', 'point_source_id', 'gps_time',
                 'red', 'green', 'blue', 'nir')
    src_names = set(src.point_format.dimension_names)
    dst_names = set(out.point_format.dimension_names)
    for d in copy_dims:
        if d in src_names and d in dst_names:
            try:
                setattr(out, d, getattr(src, d))
            except Exception:
                pass

    # Original classification preserved, new classification written.
    out.gt_class = np.array(src.classification, dtype=np.uint8)
    out.classification = cls_new.astype(np.uint8)
    out.dtm_pred_z = dtm_pred_z.astype(np.float32)
    out.residual_z = residual_z.astype(np.float32)
    out.prob_ground = prob_ground_u16.astype(np.uint16)

    out_laz.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(out_laz))


def infer_one_scene(raster_path: Path, laz_path: Path, ckpt_path: Path,
                     out_laz: Path, *,
                     alpha: float = 0.20,
                     coarse_size: int = 256,
                     tile_size: int = 256,
                     overlap: int = 128,
                     blend_mode: str = 'linear',
                     device: str = None,
                     use_priostitch: bool = True,
                     tta: int = 8) -> dict:
    """Full inference pipeline for one scene. Returns stats dict.

    Defaults follow paper §3.3 + §8 Table 7 exactly.

    `tta`: number of D2/D4 test-time-augmentation passes per tile.
    Default 8 (full D4). Each prediction is averaged across all
    transforms. 8x slower per tile but reduces aliasing and improves
    accuracy on the final test outputs (matches stage 1's default).
    """
    import laspy

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)

    # 1. Load preprocessed raster.
    with np.load(str(raster_path)) as z:
        dsm_max = z['dsm_max']
        dsm_min = z['dsm_min']
        dsm_mean = z['dsm_mean']
        dsm_mask = z['dsm_mask'].astype(np.float32)
        bbox = tuple(z['bbox'].tolist())
        gsd = float(z['gsd'])
        alpha_built = float(z['alpha_metres'])
    if abs(alpha_built - alpha) > 1e-6:
        print(f"  [warn] α_metres on raster ({alpha_built}) != requested ({alpha}); "
              f"per-pixel classification uses the new α at point-level.")

    # 2. Load checkpoint + run inference.
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    # Strip _doc / _comment style config-file keys so they don't get
    # forwarded as kwargs to the model constructor.
    def _strip_doc(d):
        return {k: v for k, v in d.items() if not k.startswith('_')} \
            if isinstance(d, dict) else d
    model = GrounDiffRaster(
        backbone=str(cfg.get('backbone', 'unet')),
        backbone_kwargs=_strip_doc(cfg.get('model', {})),
        diffusion_kwargs=_strip_doc(cfg.get('diffusion', {})),
        loss_kwargs=_strip_doc(cfg.get('loss', {})),
    ).to(device).eval()

    # Prefer EMA weights if present.
    if 'ema' in ckpt:
        shadow = ckpt['ema']['shadow']
        sd = model.state_dict()
        for k, v in shadow.items():
            if k in sd:
                sd[k].copy_(v.to(device))
        print("  [ok] loaded EMA weights")
    else:
        # New checkpoints have `net`; pre-UNet ones used `dit`. Same
        # state-dict either way.
        weights = ckpt.get('net', ckpt.get('dit'))
        if weights is None:
            raise KeyError(
                "checkpoint has neither 'net' nor 'dit' state-dict key")
        model.net.load_state_dict(weights)
        print("  [ok] loaded raw weights (no EMA in ckpt)")

    print(f"  [info] running PrioStitch inference "
          f"(coarse={coarse_size}, tile={tile_size}, overlap={overlap}, "
          f"blend={blend_mode!r}, use_priostitch={use_priostitch}, tta={tta})")
    n_done = {'v': 0}
    def cb(done, tot):
        n_done['v'] = done
        if done % max(1, tot // 20) == 0:
            print(f"    {done}/{tot} tiles")

    t0 = time.time()
    res = priostitch_infer(
        model, dsm_max, dsm_min, dsm_mean, dsm_mask,
        coarse_size=coarse_size, tile_size=tile_size, overlap=overlap,
        blend_mode=blend_mode,
        device=device, use_priostitch=use_priostitch,
        tta=tta, progress_cb=cb)
    t_inf = time.time() - t0
    print(f"  [info] inference: {t_inf:.1f}s ({n_done['v']} tiles, tta={tta})")

    dtm_pred = res['dtm_pred']            # (H, W) metres
    prob_g = res['prob_ground']           # (H, W) [0, 1]

    # 3. Load LAZ for per-point classification.
    with laspy.open(str(laz_path)) as f:
        las = f.read()
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    n_pts = xyz.shape[0]
    print(f"  [info] LAZ has {n_pts:,} points")

    # 4. Per-point: bilinear sample dtm_pred at (x, y), compute residual,
    # classify as ground iff |z - ẑ| < alpha.
    dtm_at_pts = bilinear_sample(dtm_pred, xyz[:, 0], xyz[:, 1], bbox, gsd)
    residual = (xyz[:, 2].astype(np.float32) - dtm_at_pts)
    is_ground = (np.abs(residual) < alpha)
    cls_new = np.where(is_ground, 2, 1).astype(np.uint8)
    # Also sample prob_ground (for richer LAZ extra dim).
    prob_at_pts = bilinear_sample(prob_g, xyz[:, 0], xyz[:, 1], bbox, gsd)
    prob_u16 = np.clip(prob_at_pts * 65535.0, 0, 65535).astype(np.uint16)

    # 5. Per-point stats vs original LAZ classification.
    gt = np.array(las.classification, dtype=np.uint8)
    gt_ground = np.isin(gt, [2, 9])
    pred_ground = is_ground
    # Sithole-Vosselman three rates.
    n_gt_g = int(gt_ground.sum())
    n_gt_ng = int((~gt_ground).sum())
    n_t1 = int((gt_ground & (~pred_ground)).sum())  # type-I = predicted ng but is ground
    n_t2 = int(((~gt_ground) & pred_ground).sum())  # type-II = predicted ground but is ng
    e_t1 = 100.0 * n_t1 / max(n_gt_g, 1)
    e_t2 = 100.0 * n_t2 / max(n_gt_ng, 1)
    e_tot = 100.0 * (n_t1 + n_t2) / n_pts

    print(f"  [stats] per-point vs LAZ cls∈{{2,9}}:")
    print(f"          E_T1 = {e_t1:6.2f}%   ({n_t1:,} of {n_gt_g:,} ground points)")
    print(f"          E_T2 = {e_t2:6.2f}%   ({n_t2:,} of {n_gt_ng:,} non-ground points)")
    print(f"          E_tot= {e_tot:6.2f}%   ({n_t1 + n_t2:,} of {n_pts:,} total)")

    # 6. Write output LAZ.
    print(f"  [info] writing {out_laz}")
    _write_laz_with_classification(
        laz_path, out_laz,
        cls_new=cls_new,
        dtm_pred_z=dtm_at_pts,
        residual_z=residual,
        prob_ground_u16=prob_u16)

    return dict(
        scene=raster_path.parent.name,
        n_pts=int(n_pts),
        n_gt_g=n_gt_g, n_gt_ng=n_gt_ng,
        n_t1=n_t1, n_t2=n_t2,
        e_t1_pct=e_t1, e_t2_pct=e_t2, e_tot_pct=e_tot,
        inference_s=t_inf,
        out_laz=str(out_laz),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raster', required=True, help="Path to raster.npz")
    ap.add_argument('--laz', required=True, help="Path to input LAZ")
    ap.add_argument('--ckpt', required=True, help="Path to model checkpoint")
    ap.add_argument('--out', required=True, help="Path for output LAZ")
    ap.add_argument('--alpha', type=float, default=0.20)
    # Paper-faithful PrioStitch defaults per arxiv §3.3 + §8 Table 7,
    # scaled for our 512² training (DEFRA at 0.1 m/px, with paper-256²
    # tile size deviation for context window — see configs/defra.json):
    # - coarse_size = network's input dimensions = 256
    # - tile_size   = 256 (matches training, paper §7.1)
    # - overlap     = 128 (paper's 50%-overlap, stride 128)
    # - blend       = 'linear' (paper §12.2: "best balance of performance
    #                 and visual quality"; weighted-mean across overlaps
    #                 → no seams. min has marginally better RMSE but the
    #                 paper notes it "may introduce sharp elevation jumps
    #                 at tile boundaries" — visible gridding.)
    ap.add_argument('--coarse-size', type=int, default=256)
    ap.add_argument('--tile-size', type=int, default=256)
    ap.add_argument('--overlap', type=int, default=128)
    ap.add_argument('--blend', type=str, default='linear',
                    choices=['min', 'max', 'mean', 'linear',
                             'cosine', 'exponential'],
                    help="Tile-overlap blending mode (paper §12.2 Table 7). "
                         "linear (default): distance-from-edge weighted mean, "
                         "smooth seams, paper's best visual-quality mode. "
                         "min: best RMSE (Tab.7 0.514) but sharp seams. "
                         "cosine/exponential: other smooth weighted modes. "
                         "mean: simple average. max: rarely useful.")
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--no-priostitch', action='store_true',
                    help="Disable PrioStitch (use noisy_dsm init only).")
    ap.add_argument('--tta', type=int, default=8, choices=[1, 4, 8],
                    help="Test-time augmentation passes per tile. "
                         "1 = off, 4 = D2 (flips), 8 = D4 (flips+rot). "
                         "Default 8 (full D4). Costs `tta`x inference time.")
    args = ap.parse_args()

    stats = infer_one_scene(
        Path(args.raster), Path(args.laz), Path(args.ckpt), Path(args.out),
        alpha=args.alpha,
        coarse_size=args.coarse_size,
        tile_size=args.tile_size,
        overlap=args.overlap,
        blend_mode=args.blend,
        device=args.device,
        use_priostitch=not args.no_priostitch,
        tta=args.tta,
    )
    stats_json = Path(args.out).with_suffix('.stats.json')
    stats_json.write_text(json.dumps(stats, indent=2))
    print(f"\nStats written to {stats_json}")


if __name__ == '__main__':
    main()

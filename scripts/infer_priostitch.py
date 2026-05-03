"""PrioStitch full-scene inference for a single normalised DSM.

This is the production inference path: given a checkpoint and a
preprocessed scene `.npz` (or a raw raster), produces the predicted
DTM at the original resolution.

Usage:
  python -u -m scripts.infer_priostitch \
      --config configs/defra.json \
      --resume <ckpt.pt> \
      --scene_npz /data/england_tiles/test/EN_TQ24/_scene_EN_TQ24.npz \
      --out_path /tmp/EN_TQ24_pred_dtm.npy
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.normalize import denormalise          # noqa: E402
from models import GrounDiff                    # noqa: E402
from utils import priostitch_inference          # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume', required=True)
    p.add_argument('--scene_npz', required=True,
                   help="A _scene_*.npz produced by preprocess.py")
    p.add_argument('--out_path', required=True,
                   help="Where to save the predicted DTM (.npy or .npz).")
    p.add_argument('--blend_mode', default='min')
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--init_prior', action='store_true', default=True)
    p.add_argument('--no_init_prior', dest='init_prior', action='store_false')
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GrounDiff(
        unet_kwargs=cfg['model']['unet'],
        diffusion_kwargs=cfg['model']['diffusion'],
        loss_kwargs=cfg['model'].get('loss', {}),
    ).to(device)
    d = torch.load(args.resume, map_location=device, weights_only=False)
    model.unet.load_state_dict(d['unet'] if 'unet' in d else d)
    model.eval()

    with np.load(args.scene_npz, allow_pickle=True) as f:
        if not all(k in f.files for k in ('dsm', 'valid')):
            raise RuntimeError(
                f"scene file {args.scene_npz} missing required keys "
                f"(found {list(f.files)})")
        dsm = f['dsm'].astype(np.float32)
        valid = f['valid'].astype(bool)
        stats = f['stats'].item() if 'stats' in f.files else {}
        bbox = f['bbox'] if 'bbox' in f.files else None
        gsd = float(f['gsd']) if 'gsd' in f.files else None
        dsm_min = (f['dsm_min'].astype(np.float32)
                   if 'dsm_min' in f.files else None)

    if not stats.get('has_data', True):
        raise RuntimeError(
            f"scene {args.scene_npz} has has_data=False, cannot run "
            f"PrioStitch inference")

    # PrioStitch handles per-tile DSM-only normalisation internally
    # and returns DTM in METRES.
    pred_m = priostitch_inference(
        model, dsm_full=np.zeros_like(dsm),  # unused
        dsm_metres=dsm, scene_stats=stats,
        dsm_min_metres=dsm_min,
        device=device,
        tile=cfg['model']['unet']['image_size'],
        stride=args.stride, blend_mode=args.blend_mode,
        init_prior=args.init_prior, valid_mask=valid,
        progress=lambda d, t: print(f"  tile {d}/{t}", end='\r'),
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == '.npy':
        np.save(out_path, pred_m)
    else:
        np.savez_compressed(
            out_path,
            dtm_pred=pred_m.astype(np.float32),
            valid=valid.astype(np.float32),
            bbox=bbox if bbox is not None else np.zeros(4),
            gsd=np.float32(gsd or 0.0),
            stats=stats,
        )
    print(f"\nwrote {out_path}  shape={pred_m.shape}  "
          f"valid_mean={valid.mean():.1%}")


if __name__ == '__main__':
    main()

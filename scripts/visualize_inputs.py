"""Visualize preprocessed tiles to sanity-check the pipeline.

For each sampled tile, writes a 1×3 panel:
    [normalised DSM]  [normalised DTM target]  [valid mask]

Usage:
  python -u -m scripts.visualize_inputs \
      --tile_dir /data/england_tiles \
      --out_dir runs/input_viz \
      --split test --max_tiles 16
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tile_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--max_tiles', type=int, default=16)
    args = p.parse_args()

    tile_dir = Path(args.tile_dir) / args.split
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in tile_dir.rglob('*.npz')
                   if not p.name.startswith('_scene_'))
    if not paths:
        raise SystemExit(f"no tiles found under {tile_dir}")

    rng = np.random.default_rng(0)
    chosen = rng.choice(len(paths), size=min(args.max_tiles, len(paths)),
                        replace=False)
    saved = 0
    for k, idx in enumerate(chosen):
        path = paths[int(idx)]
        try:
            with np.load(str(path), allow_pickle=True) as f:
                if not all(k in f.files
                           for k in ('dsm_norm', 'dtm_norm', 'valid_mask')):
                    print(f"  skip {path.name}: missing required keys")
                    continue
                dsm = f['dsm_norm'].astype(np.float32)
                dtm = f['dtm_norm'].astype(np.float32)
                valid = f['valid_mask'].astype(np.float32)
                stats = f['stats'].item() if 'stats' in f.files else {}
        except Exception as e:
            print(f"  skip {path.name}: {e}")
            continue

        # Mask invalid pixels for cleaner visuals
        dsm_m = np.ma.masked_where(valid < 0.5, dsm)
        dtm_m = np.ma.masked_where(valid < 0.5, dtm)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        im0 = ax[0].imshow(dsm_m, vmin=-1, vmax=1, cmap='terrain')
        ax[0].set_title(f"DSM (normed, vmin={stats.get('vmin', 0):.1f}m, "
                         f"vmax={stats.get('vmax', 0):.1f}m)")
        plt.colorbar(im0, ax=ax[0], shrink=0.8)
        im1 = ax[1].imshow(dtm_m, vmin=-1, vmax=1, cmap='terrain')
        ax[1].set_title("DTM target (normed)")
        plt.colorbar(im1, ax=ax[1], shrink=0.8)
        im2 = ax[2].imshow(valid, vmin=0, vmax=1, cmap='gray')
        ax[2].set_title(f"valid mask  ({valid.mean()*100:.1f}% covered)")
        plt.colorbar(im2, ax=ax[2], shrink=0.8)
        fig.suptitle(f"{path.parent.name}/{path.stem}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{k:02d}_{path.stem}.png", dpi=110,
                    bbox_inches='tight')
        plt.close(fig)
        saved += 1
    print(f"wrote {saved} viz panels to {out_dir}")


if __name__ == '__main__':
    main()

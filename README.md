# stage2_raster

Pixel-space diffusion with paper-faithful GrounDiff loss & gating, for
LAZ → DTM at 0.10 m resolution on DEFRA ALS. Backbone is a Palette-
derived UNet (~80M params) — the same architecture lineage as stage 1
GrounDiff, scaled up for 1024² tiles and 5-channel input. We chose
UNet over the more fashionable DiT on the empirical evidence from
Marigold (CVPR 2024) and its follow-ups that UNet remains SOTA for
dense pixel-aligned regression tasks.

## Quick start

```bash
# Edit paths at the top of INSTALL_AND_RUN.sh, then:
./INSTALL_AND_RUN.sh
```

This will: install deps → preprocess LAZ → pick 20 representative scenes
→ train 300k steps → run PrioStitch inference on those 20 → render
visualisations.

Outputs land under `$RUN_ROOT/`:
```
preprocessed/<scene>/raster.npz       raster cache (idempotent, .done markers)
run/train.log                         human-readable log
run/metrics.csv                       per-step  (step,lr,loss,l1,l2,grad,conf,sps)
run/val_metrics.csv                   per-eval  (step,loss,l1,l2,grad,conf,
                                        e_t1_pct,e_t2_pct,e_tot_pct,n_tiles,n_pixels)
run/split.json                        stratified train/val scene-name split
                                        (reused across resumes; delete to re-split)
run/viz_scenes.json                   the 20 representative val scenes picked
                                        before training (reused on resume)
run/viz/step_NNNNNNNN/                per-val visualisations -- one dir per val
  ├ <scene>_elevation.png             pass. Each holds elevation + analysis PNGs
  ├ <scene>_analysis.png              + a metrics CSV for every viz scene + a
  ├ <scene>_metrics.csv               rollup.csv across the whole step.
  └ rollup.csv
run/latest.pt, run/best.pt            checkpoints (resumable)
eval/laz/<scene>.classified.laz       FINAL output LAZ with extra dims
eval/laz/<scene>.classified.stats.json per-point E_T1/E_T2/E_tot vs original cls
eval/metrics_rollup.csv               final per-scene rollup across all 20 scenes
```

The `run/viz/step_NNNNNNNN/` directories let you scrub through training and
watch the same 20 held-out scenes improve: any image viewer's "browse next file"
key (or `ffmpeg -framerate 2 -pattern_type glob -i 'run/viz/*/SCENE_elevation.png'
training_progress.mp4`) gives you a flipbook.

## Design choices (paper-faithful or explicit deviation)

| Component | Source | Notes |
|---|---|---|
| Loss `L1 + L2 + 0.1·L∇ + 0.1·L_c` | GrounDiff §3.2 Eq. 11-14 | unchanged. L∇ uses gradient *magnitude* (paper §3.2). |
| Gating `σ(ℓ)·s + (1-σ(ℓ))·(s-r̂)` | GrounDiff Eq. 5 | unchanged. |
| Diffusion T=10 cosine β | GrounDiff §7.3 | unchanged. γ-encoded (Palette) for continuous training γ. |
| α_metres = 0.20 m | Sithole-Vosselman 2003 §4.2.1 | built into preprocess; M_α stored in raster.npz. |
| 5 input channels: g_t + dsm_max/min/mean/mask | new (paper uses single DSM) | min and mean help separate canopy from ground; mask flags empty cells. |
| Pixel-space UNet backbone | Palette / OpenAI-guided-diffusion lineage; same arch family as stage 1; choice validated by Marigold (CVPR 2024) | 5 levels for 1024² tile, channel_mults=(1,2,4,4,8), attention at 64×64 bottleneck, ~80M params. Conv inductive bias matches dense regression on terrain. |
| Per-tile normalisation: DSM-only frame | deviation from GrounDiff §7.2 | Paper uses min/max of DSM ∪ DTM; can't use DTM at inference. We use `[dsm_min.min(), dsm_max.max()]` with a 1 m min-span. |
| Hard mining | new | per-tile EMA loss tracker; Metropolis-style acceptance. |

## Repository layout

```
stage2_raster/
├── README.md
├── INSTALL_AND_RUN.sh           # one-shot pipeline driver
├── configs/
│   └── defra.json               # production config: 300k steps, UNet (~80M params)
├── models/
│   ├── __init__.py              # exports
│   ├── nn_unet.py               # GroupNorm32, zero_module, gamma_embedding helpers
│   ├── unet.py                  # GrounDiffUNet (Palette-derived; stage-1 lineage scaled up)
│   ├── gating.py                # paper Eq. 5
│   ├── diffusion.py             # T=10 cosine + γ-encoding + sampling
│   ├── losses.py                # paper §3.2 Eq. 11-14
│   └── groundiff_raster.py      # top-level wrapper
├── data/
│   ├── __init__.py
│   ├── preprocess.py            # LAZ → raster.npz {dsm_max/min/mean/mask, gt_dtm, valid, m_alpha, had_ground_return, bbox, gsd, alpha_metres}
│   └── dataset.py               # RasterTileDataset (iterable, hard mining) + RasterEvalTiles
├── scripts/
│   ├── train.py                 # AdamW + cosine LR + EMA, CSV logging, OOM-safe, resumable
│   ├── infer_to_laz.py          # PrioStitch → per-point bilinear sample → LAZ with extra dims
│   ├── visualize.py             # 5-panel figure
│   └── select_scenes.py         # pick 20 PrioStitch-sized scenes, diverse by pct_M_α
└── utils/
    ├── __init__.py
    ├── colormap.py              # hypsometric + BWR + classification colormaps
    └── priostitch.py            # coarse_pass + fine_pass + priostitch_infer
```

## Classification (what counts as ground)

There are two layers:

**Cell-level** (the 10×10 cm raster grid): a cell is predicted ground iff
`σ(ℓ) ≥ 0.5` where `ℓ` is the per-pixel logit from the model's confidence
head. This is what the `Classification` visualisation panel shows and what
`val_metrics.csv` E_T1/E_T2 are computed against.

**Point-level** (the original ALS points): each LAZ point at `(x, y, z)`
gets `ẑ = bilinear_sample(dtm_pred, x, y)`, then is classified as ground
iff `|z − ẑ| < α` (with α defaulting to 0.20 m, the same threshold used
to build the training mask M_α). This is what `infer_to_laz.py` writes
back to the output LAZ classification field.

The two agree by construction at trained convergence: σ(ℓ) ≈ M_α =
`|s − ĝ| < α` at the cell, and per-point classification with the same α
threshold lifts that decision from the cell to its constituent points.

## Inputs / outputs

**Input**: a directory of `.copc.laz` files (typical DEFRA ALS).

**Preprocess output** per scene `<name>/raster.npz`:
- `dsm_max`, `dsm_min`, `dsm_mean`  `[H,W]` float32, metres
- `dsm_mask`   `[H,W]` uint8, 1 iff cell has ≥1 ALS return
- `gt_dtm`     `[H,W]` float32, TIN-interpolated GT
- `valid`      `[H,W]` uint8, GT defined (inside TIN hull)
- `m_alpha`    `[H,W]` uint8, `|s − g| < α_metres` AND valid
- `had_ground_return` `[H,W]` uint8, ≥1 ALS return classified as ground (cls 2/9)
- `bbox`, `gsd`, `alpha_metres`

**Final LAZ output** has standard `classification` overwritten (2 = ground,
1 = non-ground) plus four extra dimensions:
- `dtm_pred_z`  (float32): ẑ at each point
- `residual_z`  (float32): z − ẑ
- `prob_ground` (uint16): σ(ℓ) at each point (after bilinear sample)
- `gt_class`    (uint8): the original LAZ classification, preserved

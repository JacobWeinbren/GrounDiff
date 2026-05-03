# GrounDiff (clean rebuild)

Paper-faithful reimplementation of GrounDiff (Dhaouadi et al., WACV 2026,
arXiv:2511.10391), trained on DEFRA UK LIDAR. See `DECISIONS.md` for
the small principled deviations (cls=2+9 for DTM ground returns).

The diffusion / denoising scaffolding is adapted from the Palette
image-to-image diffusion implementation (Janspiry/Palette-Image-to-Image-
Diffusion-Models), which itself adapts openai/guided-diffusion.

## Install

```bash
pip install -r requirements.txt
```

## Verified paper claims

| Claim                          | Paper      | This impl. | Where verified                          |
|--------------------------------|------------|------------|-----------------------------------------|
| UNet parameter count           | 62.6M      | 62.64M     | analytic, see `DECISIONS.md`            |
| UNet input channels            | 2          | 2          | `models/unet.py` `in_channel=2`         |
| UNet output channels           | 2 (r̂, ℓ)  | 2          | `models/unet.py` `out_channel=2`        |
| Diffusion forward Eq.2         | √ᾱ_t·g₀+√(1-ᾱ_t)·ε | same | `models/diffusion.py::q_sample`     |
| Gating Eq.5                    | σ(ℓ)·s + (1-σ(ℓ))·(s-r̂) | same | `models/diffusion.py::gating`        |
| Loss Eq.11-14, λ₁=λ₂=1, λ_∇=λ_c=0.1 | yes  | yes        | `models/losses.py::groundiff_loss`      |
| Gradient loss Eq.13 magnitude-only | yes    | yes        | `models/losses.py::_grad_magnitude`     |
| Norm Eq.15 joint min-max → [-1,1] | yes     | yes        | `data/normalize.py`                     |
| Aug §7.1 rot/jitter/resize/crop/flip @ p=0.5 | yes | yes  | `data/augment.py::groundiff_augment`    |
| Init at inference `g_T ~ N(s, I)` | yes     | yes        | `diffusion.py::sample` w/ `noisy_dsm`   |
| PrioStitch §3.3 prior-init + blend | yes    | yes        | `utils/priostitch.py`                   |
| AdamW lr=1e-4 wd=0.01 cos+warmup 500 | yes  | yes        | `scripts/train.py` + `configs/*.json`   |
| T=10, β cosine [1e-4, 2e-2]    | yes        | yes        | `models/diffusion.py::_make_betas`      |

## Workflow

### 1. Preprocess DEFRA LAZ → tiles

```bash
python -u -m scripts.preprocess \
    --laz_root /data/england_split \
    --out_dir  /data/england_tiles \
    --gsd 0.5 \
    --tile 256 \
    --workers 8
```

Expected wall-clock: 30-45 min on the full DEFRA corpus (~960 LAZ
files, 8 workers, H100). Output: `/data/england_tiles/{train,test}/`,
~200-400 GB total at fp16+gzip compression.

### 2. Sanity-check the tiles

```bash
python -u -m scripts.visualize_inputs \
    --tile_dir /data/england_tiles \
    --out_dir runs/input_viz \
    --split test --max_tiles 16
```

You should see DSM + DTM normed to [-1, 1] with a high-coverage valid
mask. Empty/0% tiles indicate a preprocessing problem (e.g.
ground-class returns missing — check `cls 2 ∪ 9` actually present).

### 3. Train

```bash
python -u -m scripts.train \
    --config configs/defra.json \
    --tile_dir /data/england_tiles \
    --name_suffix v1
```

Logs to `experiments/groundiff_defra_v1_<timestamp>/`:
- `train.log` — text log
- `tb/` — TensorBoard scalars
- `checkpoint/best.pt` — best-by-val-RMSE
- `checkpoint/epoch_NNNN.pt` — periodic snapshots

Validation runs every epoch and prints physical-unit metrics:

```
ep N val (n=...): RMSE=0.18m  MAE=0.07m  err>0.5m=2.1%  err>1.0m=0.5%
```

Paper Tab.1 baseline (DSM-only on DALES): RMSE 0.16m, MAE 0.094m,
err>0.5m 5.2%, err>1.0m 2.0%. With ~250× more training data than
DALES, expect to meet or beat these numbers within a few epochs.

### 4. Test (full PrioStitch evaluation)

```bash
python -u -m scripts.test \
    --config configs/defra.json \
    --tile_dir /data/england_tiles \
    --resume experiments/groundiff_defra_v1_<run>/checkpoint/best.pt \
    --out_dir runs/test_eval
```

Writes `runs/test_eval/metrics.csv` with per-scene + global numbers.

### 5. Single-scene inference

```bash
python -u -m scripts.infer_priostitch \
    --config configs/defra.json \
    --resume experiments/.../best.pt \
    --scene_npz /data/england_tiles/test/EN_TQ24/_scene_EN_TQ24.npz \
    --out_path /tmp/EN_TQ24_pred_dtm.npz
```

Output `.npz` contains `dtm_pred` (predicted DTM in metres), `valid`,
`bbox`, `gsd`, `stats`.

## Repo layout

```
groundiff/
├── README.md
├── DECISIONS.md
├── requirements.txt
├── configs/
│   ├── default.json     # paper-faithful defaults
│   └── defra.json       # DEFRA path overrides
├── data/
│   ├── normalize.py     # paper Eq.15 joint min-max
│   ├── augment.py       # paper §7.1 augmentation
│   └── dataset.py       # tile dataset
├── models/
│   ├── nn.py            # building blocks (GroupNorm, gamma_embedding, ...)
│   ├── unet.py          # 62.6M-param GrounDiff U-Net
│   ├── diffusion.py     # forward/reverse + gating
│   ├── losses.py        # paper Eq.11-14
│   ├── metrics.py       # RMSE, MAE, E_T thresholds
│   └── groundiff.py     # top-level wrapper
├── scripts/
│   ├── preprocess.py    # LAZ → DSM+DTM tiles
│   ├── train.py
│   ├── test.py          # PrioStitch eval + CSV
│   ├── visualize_inputs.py
│   └── infer_priostitch.py
└── utils/
    ├── checkpoint.py
    └── priostitch.py    # paper §3.3 tiling/blending
```

## Acknowledgement

- Paper: Dhaouadi, Meier, Kaiser, Cremers. *GrounDiff: Diffusion-Based
  Ground Surface Generation from Digital Surface Models.* WACV 2026.
- Diffusion scaffolding adapted from
  [Palette-Image-to-Image-Diffusion-Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
  which adapts [openai/guided-diffusion](https://github.com/openai/guided-diffusion).

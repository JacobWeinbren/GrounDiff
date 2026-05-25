#!/usr/bin/env bash
# stage2_raster: install dependencies and run the full pipeline.
#
# Steps:
#   1) Install Python deps (laspy[laszip], scipy, matplotlib, Pillow, torch).
#   2) Preprocess LAZ scenes → raster.npz cache.
#   3) Pick 20 representative PrioStitch-sized scenes for the eval set.
#   4) Train (60 epochs) with CSV logging.
#   5) Run PrioStitch inference on the 20 chosen scenes → output LAZ + .npz.
#   6) Render 20 visualisations.
#
# Configurable paths at the top. Each step is independently re-runnable:
# the preprocessor uses .done markers, training resumes from latest.pt, and
# inference + visualization skip files that already exist (with --force to
# override).

set -euo pipefail

# -------------------------- CONFIG ----------------------------------------- #
# Where the input LAZ files are. Change this for your environment.
LAZ_DIR="${LAZ_DIR:-/root/data/data/england_raw}"

# Where to write everything.
RUN_ROOT="${RUN_ROOT:-/root/work/stage2_raster_runs}"

# Stage 2 raster code dir (containing models/, data/, scripts/, configs/).
CODE_DIR="${CODE_DIR:-$(dirname "$(readlink -f "$0")")}"

# Config to use.
CONFIG="${CONFIG:-${CODE_DIR}/configs/defra.json}"

# Derived paths.
PREP_DIR="${RUN_ROOT}/preprocessed"
RUN_DIR="${RUN_ROOT}/run"
EVAL_DIR="${RUN_ROOT}/eval"
LAZ_OUT="${EVAL_DIR}/laz"
VIZ_OUT="${EVAL_DIR}/viz"
SCENES_JSON="${EVAL_DIR}/representative_scenes.json"

mkdir -p "${RUN_ROOT}" "${PREP_DIR}" "${RUN_DIR}" "${EVAL_DIR}" \
         "${LAZ_OUT}" "${VIZ_OUT}"

export PYTHONPATH="${CODE_DIR%/*}:${PYTHONPATH:-}"

# Auto-detect which python.
PY="${PY:-python3}"

log() { echo "[$(date +'%H:%M:%S')] $*"; }

# -------------------------- 1) INSTALL ------------------------------------- #
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  log "Installing Python dependencies..."
  ${PY} -m pip install --break-system-packages --quiet --upgrade pip
  ${PY} -m pip install --break-system-packages --quiet \
        'laspy[laszip]>=2.5' scipy>=1.10 matplotlib>=3.7 Pillow>=10.0 numpy tqdm
  if ! ${PY} -c 'import torch' 2>/dev/null; then
    log "PyTorch not installed; please install a CUDA-matching build first."
    log "  e.g. pip install torch --index-url https://download.pytorch.org/whl/cu128"
    exit 1
  fi
  log "  torch: $(${PY} -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
fi

# -------------------------- 2) PREPROCESS ---------------------------------- #
# How many parallel preprocessing workers. Each loads one LAZ at a time and
# can use ~0.5-2 GB RAM. Default: half of CPU count, capped at 16. Override:
#   PREP_WORKERS=24 ./INSTALL_AND_RUN.sh
PREP_WORKERS="${PREP_WORKERS:-0}"

log "Preprocessing LAZ → raster.npz (parallel, idempotent via .done markers)..."
${PY} -m stage2_raster.data.preprocess \
    --laz_dir "${LAZ_DIR}" \
    --out_dir "${PREP_DIR}" \
    --gsd 0.10 \
    --alpha_metres 0.20 \
    --workers "${PREP_WORKERS}"

# -------------------------- 3) PRETILE ------------------------------------- #
# Slice each preprocessed scene into individual 256x256 tile .npz files for
# fast random access at training time. At fine GSDs (0.1 m/px) each
# preprocessed scene is multi-GB; the original scene-based dataloader stalls
# every ~256 tiles to decompress the next scene. Pretiling eliminates this.
#
# Idempotent: skips scenes that already have a .done marker in the output.
# Re-run after preprocess to add new scenes.
PRETILE_DIR="${PRETILE_DIR:-${RUN_ROOT}/pretiled}"
PRETILE_WORKERS="${PRETILE_WORKERS:-16}"
log "Pretiling preprocessed scenes → 256x256 tile .npz files..."
${PY} -m stage2_raster.scripts.pretile \
    --input_root "${PREP_DIR}" \
    --output_root "${PRETILE_DIR}" \
    --tile_size 256 --stride 256 \
    --num_workers "${PRETILE_WORKERS}"

# -------------------------- 4) TRAIN --------------------------------------- #
# Training reads from PRETILE_DIR (per-tile .npz files), set in
# configs/defra.json under data.pretile_root. Val still reads from PREP_DIR
# scene caches (val runs infrequently, RasterEvalTiles handles it).
# Training is run BEFORE scene selection so that train.py creates the
# stratified train/val split (written to <out_dir>/split.json). The scene
# picker then restricts its representative-20 picks to val scenes only,
# guaranteeing the model never saw them at training time.
log "Training (60 epochs, see ${RUN_DIR}/train.log)..."
${PY} -m stage2_raster.scripts.train \
    --config "${CONFIG}" \
    --out_dir "${RUN_DIR}" \
    --train_root "${PREP_DIR}" \
    --val_root "${PREP_DIR}"

# -------------------------- 5) SELECT 20 SCENES ---------------------------- #
# Restrict picks to the held-out val scenes, so visualisations show how
# the model performs on unseen data (not the cached training fit).
log "Selecting 20 representative val scenes for PrioStitch viz..."
${PY} -m stage2_raster.scripts.select_scenes \
    --root "${PREP_DIR}" \
    --tile-size 1024 \
    --min-factor 2.0 \
    --n 20 \
    --split-json "${RUN_DIR}/split.json" \
    --out "${SCENES_JSON}"

# -------------------------- 6) INFERENCE ----------------------------------- #
log "Running PrioStitch inference on the 20 selected scenes..."
CKPT="${RUN_DIR}/best.pt"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="${RUN_DIR}/latest.pt"
fi
if [[ ! -f "${CKPT}" ]]; then
  log "ERROR: no checkpoint found in ${RUN_DIR}"
  exit 1
fi
log "  using ${CKPT}"

# Read scene names out of the JSON.
SCENES=$(${PY} -c "import json; d=json.load(open('${SCENES_JSON}')); print(' '.join(s['name'] for s in d['scenes']))")

for SCENE in ${SCENES}; do
  RASTER="${PREP_DIR}/${SCENE}/raster.npz"
  # Find the input LAZ for this scene (preprocess strips '.copc' from the stem).
  LAZ_IN=""
  for ext in '.copc.laz' '.laz'; do
    cand="${LAZ_DIR}/${SCENE}${ext}"
    if [[ -f "${cand}" ]]; then
      LAZ_IN="${cand}"; break
    fi
  done
  if [[ -z "${LAZ_IN}" ]]; then
    log "  [warn] no LAZ found for scene ${SCENE} (looked under ${LAZ_DIR})"
    continue
  fi
  LAZ_OUT_F="${LAZ_OUT}/${SCENE}.classified.laz"
  if [[ -f "${LAZ_OUT_F}" && "${FORCE:-0}" != "1" ]]; then
    log "  [skip] ${SCENE} (output exists)"
    continue
  fi
  log "  ${SCENE}: ${LAZ_IN} → ${LAZ_OUT_F}"
  ${PY} -m stage2_raster.scripts.infer_to_laz \
      --raster "${RASTER}" \
      --laz "${LAZ_IN}" \
      --ckpt "${CKPT}" \
      --out "${LAZ_OUT_F}"
done

# -------------------------- 7) VISUALIZE ----------------------------------- #
log "Rendering 20 visualisations (elevation + analysis + metrics CSV each)..."
ROLLUP_CSV="${EVAL_DIR}/metrics_rollup.csv"
# Wipe the rollup so reruns start fresh; per-scene CSVs are still written.
: > "${ROLLUP_CSV}"
for SCENE in ${SCENES}; do
  RASTER="${PREP_DIR}/${SCENE}/raster.npz"
  ELEV_PNG="${VIZ_OUT}/${SCENE}_elevation.png"
  if [[ -f "${ELEV_PNG}" && "${FORCE:-0}" != "1" ]]; then
    log "  [skip] ${SCENE} viz (exists)"
    continue
  fi
  log "  ${SCENE}: → ${VIZ_OUT}/${SCENE}_{elevation,analysis}.png + _metrics.csv"
  ${PY} -c "
import sys
sys.path.insert(0, '${CODE_DIR%/*}')
import numpy as np
import torch
from pathlib import Path
from stage2_raster.models import GrounDiffRaster
from stage2_raster.utils.priostitch import priostitch_infer
from stage2_raster.scripts.visualize import make_all

with np.load('${RASTER}') as z:
    dsm_max = z['dsm_max']; dsm_min = z['dsm_min']; dsm_mean = z['dsm_mean']
    dsm_mask = z['dsm_mask']; gt_dtm = z['gt_dtm']; valid = z['valid']
    had_g = z.get('had_ground_return', None)
    gsd = float(z['gsd']); alpha = float(z['alpha_metres'])

ckpt = torch.load('${CKPT}', map_location='cpu', weights_only=False)
cfg = ckpt.get('config', {})
m = GrounDiffRaster(backbone=str(cfg.get('backbone', 'unet')),
                     backbone_kwargs={k:v for k,v in cfg.get('model', {}).items() if not k.startswith('_')},
                     diffusion_kwargs={k:v for k,v in cfg.get('diffusion', {}).items() if not k.startswith('_')},
                     loss_kwargs={k:v for k,v in cfg.get('loss', {}).items() if not k.startswith('_')})
_sd = ckpt.get('net', ckpt.get('dit'))
m.net.load_state_dict(_sd)
if 'ema' in ckpt:
    sd = m.state_dict()
    for k, v in ckpt['ema']['shadow'].items():
        if k in sd: sd[k].copy_(v)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
m = m.to(device).eval()

res = priostitch_infer(m, dsm_max, dsm_min, dsm_mean, dsm_mask.astype(np.float32),
    coarse_size=1024, tile_size=1024, overlap=256, device=device, tta=8)

make_all(
    dsm_max=dsm_max, gt_dtm=gt_dtm,
    valid_gt=valid, dsm_mask=dsm_mask,
    dtm_pred=res['dtm_pred'],
    had_ground=had_g, logit=res['logit'],
    gsd=gsd, alpha_metres=alpha,
    title_prefix='stage2_raster', err_vrange=1.0,
    out_prefix=Path('${VIZ_OUT}') / '${SCENE}',
    dpi=200, scene_name='${SCENE}',
    rollup_csv=Path('${ROLLUP_CSV}'))
"
done

log "All done."
log "  Logs:    ${RUN_DIR}/train.log"
log "  CSVs:    ${RUN_DIR}/metrics.csv, ${RUN_DIR}/val_metrics.csv"
log "  LAZs:    ${LAZ_OUT}/"
log "  Viz:     ${VIZ_OUT}/<scene>_elevation.png + _analysis.png + _metrics.csv"
log "  Rollup:  ${ROLLUP_CSV}"
log "  Best:    ${RUN_DIR}/best.pt"

#!/usr/bin/env bash
# GrounDiff + Noise2Noise on multi-year EA surveys, end to end:
# random 5 km squares of England with repeat surveys -> download each year's point cloud + EA DTM ->
# rasters on one grid per place -> survey offsets, change masks, consensus DTM -> split ->
# diffusion training (cross-year targets) -> single-step fine-tune with the noise head ->
# test against the consensus -> ONNX export -> device speed test -> QGIS plugin zip.
#
#   caffeinate -dimsu nohup bash tools/run_n2n.sh > runs/n2n.log 2>&1 &
#
# Resumable: run the same command again after a stop; finished steps are skipped (markers in
# data/mt/done/), finished squares / scenes are kept, training continues from last.pt.
# Settings (environment): TARGET=60 squares of 5 km, CROPS=2 places (2 km) per square, MIN_YEARS=3
# surveys per square (3+ gives every place a consensus DTM), YEARS="2017 2022", FETCH_WORKERS=2,
# WORKERS=3 rasterising processes, STEPS=20000 diffusion steps, MIN_FREE_GB (default: estimated),
# STOP_AFTER=split for the data steps only.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

TARGET=${TARGET:-60}
CROPS=${CROPS:-2}
MIN_YEARS=${MIN_YEARS:-3}
YEARS=${YEARS:-"2017 2022"}
FETCH_WORKERS=${FETCH_WORKERS:-2}
WORKERS=${WORKERS:-3}
STEPS=${STEPS:-20000}
D=data/mt
mkdir -p "$D/done" runs results models

say() { echo; echo "=== $(date '+%a %H:%M') $*"; }
done_() { [ -f "$D/done/$1" ]; }
mark() { touch "$D/done/$1"; }

# free space for the steps left: ~0.6 GB per place-year of clipped lidar + DTM + rasters (at most 6 years),
# plus ~4 GB for one square's point-cloud zip being clipped at a time per fetch worker
free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
need_gb=10
if ! done_ rasterise; then
  need_gb=$((need_gb + TARGET * CROPS * 6 * 6 / 10 + FETCH_WORKERS * 4))
fi
if [ -n "${MIN_FREE_GB:-}" ]; then need_gb=$MIN_FREE_GB; fi
say "GrounDiff N2N: ${TARGET} squares x ${CROPS} places, >= ${MIN_YEARS} surveys in ${YEARS}," \
    "${free_gb} GB free, about ${need_gb} GB needed, $(git rev-parse --short HEAD)"
if [ "$free_gb" -lt "$need_gb" ]; then
  echo "Not enough free space: ${free_gb} GB free, about ${need_gb} GB needed. Free some space, lower TARGET,"
  echo "or set MIN_FREE_GB to the space you know is needed."
  exit 1
fi

if ! done_ plan; then
  say "1/11 choosing random 5 km squares with repeat surveys (EA survey catalogue)"
  # shellcheck disable=SC2086
  python -m groundiff.data.multiyear plan --out "$D" --target "$TARGET" --years $YEARS \
    --min-years "$MIN_YEARS" --crops "$CROPS"
  mark plan
fi

if ! done_ fetch; then
  say "2/11 downloading every year's point cloud (clipped to the places, zips deleted) and EA DTM"
  python -m groundiff.data.multiyear fetch --out "$D" --workers "$FETCH_WORKERS" \
    || python -m groundiff.data.multiyear fetch --out "$D" --workers 1 \
    || echo "[warn] some downloads failed twice; carrying on without them"
  mark fetch
fi

if ! done_ rasterise; then
  say "3/11 rasterising every place and year on one grid"
  python -m groundiff.data.multiyear rasterise --out "$D" --workers "$WORKERS" \
    || echo "[warn] some scenes failed (listed above); training uses the rest"
  mark rasterise
fi

if ! done_ pairs; then
  say "4/11 survey offsets, change masks, consensus DTMs, label-vs-consensus maps"
  python -m groundiff.data.multiyear pairs --out "$D"
  mark pairs
fi

if [ ! -f "$D/split.json" ]; then
  say "5/11 train / val / test split (10 km blocks; every year of a place in the same split)"
  python -m groundiff.data.split --root "$D/scenes" --out "$D/split.json"
fi
if [ "${STOP_AFTER:-}" = split ]; then
  say "data ready ($(ls "$D/scenes" | wc -l | tr -d ' ') scenes); stopping before training (STOP_AFTER=split)"
  exit 0
fi

if ! done_ train; then
  say "6/11 diffusion model from scratch, Noise2Noise targets (runs/mt; watch: python -m groundiff.monitor runs/mt --follow)"
  python -m groundiff.train configs/n2n.json --set train.num_workers=6 optim.total_steps="$STEPS"
  mark train
fi

if ! done_ train_1step; then
  say "7/11 single-step fine-tune with the label-noise head (runs/mt_1step)"
  if [ -f runs/mt_1step/last.pt ]; then
    python -m groundiff.train configs/n2n_1step.json --set train.num_workers=6
  else
    python -m groundiff.train configs/n2n_1step.json --init-from runs/mt/best.pt --set train.num_workers=6
  fi
  mark train_1step
fi

if ! done_ test; then
  say "8/11 test set: against each year's label and against the multi-year consensus"
  python -m groundiff.infer --checkpoint runs/mt_1step/best.pt --scenes "$D/scenes" \
    --split-file "$D/split.json" --split test --out results/mt_1step --no-overlays
  if [ -f runs/no_lastools_1step/best.pt ]; then
    say "   same test scenes with the earlier single-step model, for comparison"
    python -m groundiff.infer --checkpoint runs/no_lastools_1step/best.pt --scenes "$D/scenes" \
      --split-file "$D/split.json" --split test --out results/mt_old_model --no-overlays \
      || echo "[warn] the earlier model could not run on these scenes (different inputs); skipped"
  fi
  mark test
fi

say "9/11 ONNX export"
python -m groundiff.export runs/mt_1step/best.pt --out models/mt_1step
say "10/11 device speed test"
python -m groundiff.speedtest models/mt_1step.onnx --batch 8 16 || echo "[warn] speed test failed"

say "11/11 QGIS plugin zip"
python tools/build_qgis_plugin.py

say "finished. Model for QGIS: models/mt_1step.onnx   Plugin: dist/groundiff_qgis.zip"
echo "Test results: results/mt_1step/summary.json ('against the multi-year consensus' line above)"

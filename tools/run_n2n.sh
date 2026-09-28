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
# data/mt/done/), finished scenes are kept (a square's points are deleted once its scenes exist), training continues from last.pt.
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

# free space for the steps left: ~110 MB of rasters per place and year (at most 6 years), plus per
# square in flight a point-cloud zip (up to ~3 GB) and the clipped points and DTMs of its years (~6 GB)
free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
need_gb=10
if ! done_ build; then
  need_gb=$((need_gb + TARGET * CROPS * 6 * 110 / 1024 + FETCH_WORKERS * 9))
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
  say "1/10 choosing random 5 km squares with repeat surveys (EA survey catalogue)"
  # shellcheck disable=SC2086
  python -m groundiff.data.multiyear plan --out "$D" --target "$TARGET" --years $YEARS \
    --min-years "$MIN_YEARS" --crops "$CROPS"
  mark plan
fi

if ! done_ build; then
  say "2/10 per square: download each year's point cloud (clipped to the places) and EA DTM, rasterise"
  echo "     them on one grid per place, then delete that square's points (the scenes stay)"
  python -m groundiff.data.multiyear build --out "$D" --fetch-workers "$FETCH_WORKERS" --workers "$WORKERS" \
    || python -m groundiff.data.multiyear build --out "$D" --fetch-workers 1 --workers "$WORKERS" \
    || echo "[warn] some square-years failed twice (listed above); training uses the rest"
  mark build
fi

if ! done_ pairs; then
  say "3/10 survey offsets, change masks, consensus DTMs, label-vs-consensus maps"
  python -m groundiff.data.multiyear pairs --out "$D"
  mark pairs
fi

if [ ! -f "$D/split.json" ]; then
  say "4/10 train / val / test split (10 km blocks; every year of a place in the same split)"
  python -m groundiff.data.split --root "$D/scenes" --out "$D/split.json"
fi
if [ "${STOP_AFTER:-}" = split ]; then
  say "data ready ($(ls "$D/scenes" | wc -l | tr -d ' ') scenes); stopping before training (STOP_AFTER=split)"
  exit 0
fi

if ! done_ train; then
  say "5/10 diffusion model from scratch, Noise2Noise targets (runs/mt; watch: python -m groundiff.monitor runs/mt --follow)"
  python -m groundiff.train configs/n2n.json --set train.num_workers=6 optim.total_steps="$STEPS"
  mark train
fi

if ! done_ train_1step; then
  say "6/10 single-step fine-tune with the label-noise head (runs/mt_1step)"
  if [ -f runs/mt_1step/last.pt ]; then
    python -m groundiff.train configs/n2n_1step.json --set train.num_workers=6
  else
    python -m groundiff.train configs/n2n_1step.json --init-from runs/mt/best.pt --set train.num_workers=6
  fi
  mark train_1step
fi

if ! done_ test; then
  say "7/10 test set: against each year's label and against the multi-year consensus"
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

say "8/10 ONNX export"
python -m groundiff.export runs/mt_1step/best.pt --out models/mt_1step
say "9/10 device speed test"
python -m groundiff.speedtest models/mt_1step.onnx --batch 8 16 || echo "[warn] speed test failed"

say "10/10 QGIS plugin zip"
python tools/build_qgis_plugin.py

say "finished. Model for QGIS: models/mt_1step.onnx   Plugin: dist/groundiff_qgis.zip"
echo "Test results: results/mt_1step/summary.json ('against the multi-year consensus' line above)"

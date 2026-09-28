#!/usr/bin/env bash
# GrounDiff + E2E-FT trained the SAR2SAR way (Dalsasso et al. 2021: Noise2Noise across acquisition
# dates, with change compensation) on multi-year EA surveys, end to end:
# random 5 km squares of England with repeat surveys -> download each year's point cloud + EA DTM ->
# rasters on one grid per place -> year pairs + consensus DTM (evaluation) -> split ->
# step A: GrounDiff diffusion + E2E-FT single-step on each year's own label ->
# pre-estimates x_hat_a -> step B: fine-tune on compensated pairs (label_b - x_hat_b + x_hat_a, L1) ->
# pre-estimates x_hat_b -> step C: the same with x_hat_b -> test -> ONNX -> speed test -> QGIS plugin.
#
#   caffeinate -dimsu nohup bash tools/run_n2n.sh > runs/n2n.log 2>&1 &
#
# Resumable: run the same command again after a stop; finished steps are skipped (markers in
# data/mt/done/), finished scenes are kept, training continues from last.pt.
# Settings (environment): TARGET=60 squares of 5 km, CROPS=2 places (2 km) per square, MIN_YEARS=3
# surveys per square, YEARS="2017 2022", FETCH_WORKERS=2, WORKERS=2 rasterising processes,
# STEPS=20000 diffusion steps (step A), STEPS_BC (default: a third of step A, as SAR2SAR's 10 of 30
# epochs), MIN_FREE_GB (default: estimated), STOP_AFTER=split for the data steps only.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

TARGET=${TARGET:-60}
CROPS=${CROPS:-2}
MIN_YEARS=${MIN_YEARS:-3}
YEARS=${YEARS:-"2017 2022"}
FETCH_WORKERS=${FETCH_WORKERS:-2}
WORKERS=${WORKERS:-2}
STEPS=${STEPS:-20000}
STEPS_BC=${STEPS_BC:-$(( (STEPS + 3000) / 3 ))}
D=data/mt
mkdir -p "$D/done" runs results models

say() { echo; echo "=== $(date '+%a %H:%M') $*"; }
done_() { [ -f "$D/done/$1" ]; }
mark() { touch "$D/done/$1"; }

# free space for the steps left: ~110 MB of rasters per place and year (at most 6 years), plus per
# square in flight a point-cloud zip (up to ~3 GB) and its years' points clipped to 4 candidate crops (~12 GB)
free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
need_gb=10
if ! done_ build; then
  need_gb=$((need_gb + TARGET * CROPS * 6 * 110 / 1024 + FETCH_WORKERS * 15))
fi
done_ xhat_b || need_gb=$((need_gb + 12))        # two pre-estimate rasters per scene (float32)
if [ -n "${MIN_FREE_GB:-}" ]; then need_gb=$MIN_FREE_GB; fi
say "GrounDiff N2N: ${TARGET} squares x ${CROPS} places, >= ${MIN_YEARS} surveys in ${YEARS}," \
    "${free_gb} GB free, about ${need_gb} GB needed, $(git rev-parse --short HEAD)"
if [ "$free_gb" -lt "$need_gb" ]; then
  echo "Not enough free space: ${free_gb} GB free, about ${need_gb} GB needed. Free some space, lower TARGET,"
  echo "or set MIN_FREE_GB to the space you know is needed."
  exit 1
fi

if ! done_ plan; then
  say "1/12 choosing random 5 km squares with repeat surveys (EA survey catalogue)"
  # shellcheck disable=SC2086
  python -m groundiff.data.multiyear plan --out "$D" --target "$TARGET" --years $YEARS \
    --min-years "$MIN_YEARS" --crops "$CROPS"
  mark plan
fi

if ! done_ build; then
  say "2/12 per square: download each year's point cloud (clipped to the places) and EA DTM, rasterise"
  echo "     them on one grid per place, then delete that square's points (the scenes stay)"
  python -m groundiff.data.multiyear build --out "$D" --fetch-workers "$FETCH_WORKERS" --workers "$WORKERS" \
    || python -m groundiff.data.multiyear build --out "$D" --fetch-workers 1 --workers "$WORKERS" \
    || echo "[warn] some square-years failed twice (listed above); training uses the rest"
  mark build
fi

if ! done_ pairs; then
  say "3/12 survey years per place and consensus DTMs (evaluation only)"
  python -m groundiff.data.multiyear pairs --out "$D"
  mark pairs
fi

if [ ! -f "$D/split.json" ]; then
  say "4/12 train / val / test split (10 km blocks; every year of a place in the same split)"
  python -m groundiff.data.split --root "$D/scenes" --out "$D/split.json"
fi
if [ "${STOP_AFTER:-}" = split ]; then
  say "data ready ($(ls "$D/scenes" | wc -l | tr -d ' ') scenes); stopping before training (STOP_AFTER=split)"
  exit 0
fi

if ! done_ train; then
  say "5/12 step A: GrounDiff diffusion from scratch, own-year labels (runs/mt; watch: python -m groundiff.monitor runs/mt --follow)"
  python -m groundiff.train configs/n2n.json --set train.num_workers=6 optim.total_steps="$STEPS"
  mark train
fi

if ! done_ train_1step; then
  say "6/12 step A: E2E-FT single-step fine-tune, own-year labels (runs/mt_1step)"
  if [ -f runs/mt_1step/last.pt ]; then
    python -m groundiff.train configs/n2n_1step.json --set train.num_workers=6
  else
    python -m groundiff.train configs/n2n_1step.json --init-from runs/mt/best.pt --set train.num_workers=6
  fi
  mark train_1step
fi

if ! done_ xhat_a; then
  say "7/12 pre-estimates x_hat_a from step A (every scene with other years)"
  python -m groundiff.data.multiyear compensate --out "$D" --checkpoint runs/mt_1step/best.pt --name xhat_a
  mark xhat_a
fi

if ! done_ train_b; then
  say "8/12 step B: compensated year pairs, target label_b - x_hat_b + x_hat_a, ${STEPS_BC} steps (runs/mt_b)"
  if [ -f runs/mt_b/last.pt ]; then
    python -m groundiff.train configs/sar2sar.json --set train.num_workers=6 optim.total_steps="$STEPS_BC"
  else
    python -m groundiff.train configs/sar2sar.json --init-from runs/mt_1step/best.pt \
      --set train.num_workers=6 optim.total_steps="$STEPS_BC"
  fi
  mark train_b
fi

if ! done_ xhat_b; then
  say "9/12 pre-estimates x_hat_b from step B"
  python -m groundiff.data.multiyear compensate --out "$D" --checkpoint runs/mt_b/last.pt --name xhat_b
  mark xhat_b
fi

if ! done_ train_c; then
  say "10/12 step C: the same with x_hat_b, ${STEPS_BC} steps (runs/mt_c)"
  C_SET="train.num_workers=6 optim.total_steps=$STEPS_BC data.compensation=xhat_b train.out_dir=runs/mt_c"
  if [ -f runs/mt_c/last.pt ]; then
    python -m groundiff.train configs/sar2sar.json --set $C_SET
  else
    python -m groundiff.train configs/sar2sar.json --init-from runs/mt_b/last.pt --set $C_SET
  fi
  mark train_c
fi

if ! done_ test; then
  say "11/12 test set: step C and step A, against each year's label and the multi-year consensus"
  python -m groundiff.infer --checkpoint runs/mt_c/last.pt --scenes "$D/scenes" \
    --split-file "$D/split.json" --split test --out results/mt_c --no-overlays
  python -m groundiff.infer --checkpoint runs/mt_1step/best.pt --scenes "$D/scenes" \
    --split-file "$D/split.json" --split test --out results/mt_a --no-overlays
  mark test
fi

say "12/12 ONNX export, speed test, QGIS plugin"
python -m groundiff.export runs/mt_c/last.pt --out models/mt_c
python -m groundiff.speedtest models/mt_c.onnx --batch 8 16 || echo "[warn] speed test failed"
python tools/build_qgis_plugin.py

say "finished. Model for QGIS: models/mt_c.onnx   Plugin: dist/groundiff_qgis.zip"
echo "Test results: results/mt_c (SAR2SAR) and results/mt_a (step A only, i.e. GrounDiff + E2E-FT) above"

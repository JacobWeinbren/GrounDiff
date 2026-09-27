#!/usr/bin/env bash
# GrounDiff v2, end to end: stratified sample of England -> download -> EA DTM -> rasters ->
# split -> diffusion training -> single-step fine-tune -> test (per stratum, near bridges) ->
# ONNX export -> device speed test -> QGIS plugin zip.
#
#   caffeinate -dimsu nohup bash tools/run_v2.sh > runs/v2.log 2>&1 &
#
# Resumable: run the same command again after a stop; finished steps are skipped (markers in
# data/v2/done/), downloads and scenes are cached, training continues from last.pt.
# Settings (environment): TARGET=700 tiles, WORKERS=4 preprocess processes, MIN_FREE_GB=120.
set -euo pipefail
export PYTHONUNBUFFERED=1                # progress lines reach the log as they happen
cd "$(dirname "$0")/.."

TARGET=${TARGET:-700}
WORKERS=${WORKERS:-4}
MIN_FREE_GB=${MIN_FREE_GB:-120}
D=data/v2
mkdir -p "$D/done" runs results models

say() { echo; echo "=== $(date '+%a %H:%M') $*"; }
done_() { [ -f "$D/done/$1" ]; }
mark() { touch "$D/done/$1"; }

free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
used_gb=$(du -sk "$D" 2>/dev/null | awk '{print int($1 / 1048576)}')
if [ $((free_gb + used_gb)) -lt "$MIN_FREE_GB" ]; then
  echo "Only ${free_gb} GB free (+${used_gb} GB already in $D): this needs about ${MIN_FREE_GB} GB"
  echo "(laz ~20 GB, EA DTM ~30 GB, scenes ~50 GB). Free some space, lower TARGET, or set MIN_FREE_GB."
  exit 1
fi
say "GrounDiff v2: ${TARGET} tiles, ${free_gb} GB free, $(git rev-parse --short HEAD)"

if ! done_ select; then
  say "1/10 choosing tiles (lidar scan of every DEFRA 2022 tile + OpenStreetMap; ~30-60 min, resumable)"
  python -m groundiff.data.select --out "$D" --target "$TARGET" --exclude data/laz/ea
  mark select
fi

if ! done_ download; then
  say "2/10 downloading point clouds"
  python -m groundiff.data.download --out "$D/laz" --keys-file "$D/selection.json" --workers 8 \
    || python -m groundiff.data.download --out "$D/laz" --keys-file "$D/selection.json" --workers 4 \
    || echo "[warn] some tiles failed twice; carrying on without them"
  mark download
fi

if ! done_ ea_dtm; then
  say "3/10 downloading EA DTM rasters (the training target)"
  python -m groundiff.data.ea_dtm --tiles "$D/laz" --out "$D/ea_dtm" \
    || python -m groundiff.data.ea_dtm --tiles "$D/laz" --out "$D/ea_dtm" \
    || echo "[warn] some DTM tiles failed twice; those scenes are skipped"
  mark ea_dtm
fi

if ! done_ preprocess; then
  say "4/10 rasterising (bridges from OpenStreetMap, strata into meta.json)"
  python -m groundiff.data.preprocess --points-dir "$D/laz" --dtm-dir "$D/ea_dtm" --out "$D/scenes" \
    --gsd 1.0 --workers "$WORKERS" --osm "$D/osm.json" --selection "$D/selection.json" \
    || echo "[warn] some scenes failed (listed above); training uses the rest"
  mark preprocess
fi

if [ ! -f "$D/split.json" ]; then
  say "5/10 train / val / test split (10 km blocks)"
  python -m groundiff.data.split --root "$D/scenes" --out "$D/split.json"
fi

if ! done_ train; then
  say "6/10 training the diffusion model (runs/v2; watch: python -m groundiff.monitor runs/v2 --follow)"
  if [ -f runs/no_lastools/best.pt ] && [ ! -f runs/v2/last.pt ]; then
    python -m groundiff.train configs/v2.json --init-from runs/no_lastools/best.pt --set train.num_workers=6
  elif [ -f runs/v2/last.pt ] || [ -f runs/no_lastools/best.pt ]; then
    python -m groundiff.train configs/v2.json --set train.num_workers=6
  else                                   # no earlier model: from scratch, the full schedule
    python -m groundiff.train configs/v2.json --set train.num_workers=6 optim.lr=0.0001 \
      optim.warmup_steps=500 optim.total_steps=20000
  fi
  mark train
fi

if ! done_ train_1step; then
  say "7/10 single-step fine-tune (runs/v2_1step)"
  if [ -f runs/v2_1step/last.pt ]; then
    python -m groundiff.train configs/v2_1step.json --set train.num_workers=6
  else
    python -m groundiff.train configs/v2_1step.json --init-from runs/v2/best.pt --set train.num_workers=6
  fi
  mark train_1step
fi

if ! done_ test; then
  say "8/10 test set: per stratum and near bridges"
  python -m groundiff.infer --checkpoint runs/v2_1step/best.pt --scenes "$D/scenes" \
    --split-file "$D/split.json" --split test --out results/v2_1step --no-overlays
  if [ -f runs/no_lastools_1step/best.pt ]; then
    say "   same test scenes with the previous single-step model, for comparison"
    python -m groundiff.infer --checkpoint runs/no_lastools_1step/best.pt --scenes "$D/scenes" \
      --split-file "$D/split.json" --split test --out results/v2_old_model --no-overlays
  fi
  mark test
fi

say "9/10 ONNX export + device speed test"
python -m groundiff.export runs/v2_1step/best.pt --out models/v2_1step
python -m groundiff.speedtest models/v2_1step.onnx --batch 8 16 || echo "[warn] speed test failed"

say "10/10 QGIS plugin zip"
python tools/build_qgis_plugin.py

say "finished. Model for QGIS: models/v2_1step.onnx   Plugin: dist/groundiff_qgis.zip"
echo "Test results: results/v2_1step/summary.json (the tables above: new model first, then the previous one)"

#!/usr/bin/env bash
# Train the residual diffusion bridge + tile_range model on the v2 data, from scratch.
#
#   caffeinate -dimsu nohup bash tools/run_overnight_rdbm.sh > runs/rdbm.log 2>&1 &
#
# 1. finishes any v2 data steps not done yet (download, EA DTM, rasterise, split). Same commands and
#    markers (data/v2/done) as tools/run_v2.sh, which then skips them;
# 2. trains configs/v2_rdbm_range.json from scratch (20000 steps), or resumes runs/v2_rdbm_range/last.pt.
#    best.pt is written at every validation (every 1000 steps), so stopping early keeps a model;
#    run this script again to continue;
# 3. once training is done, evaluates it, the Palette-style v2 models (if trained) and the previous
#    model on the same v2 test scenes (results/cmp_*).
# Settings (environment): STEPS=20000, EVAL_SCENES=10, WORKERS=4.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
STEPS=${STEPS:-20000}
EVAL_SCENES=${EVAL_SCENES:-10}
WORKERS=${WORKERS:-4}
D=data/v2
mkdir -p "$D/done" runs results
say() { echo; echo "=== $(date '+%a %H:%M') $*"; }
done_() { [ -f "$D/done/$1" ]; }
mark() { touch "$D/done/$1"; }

[ -f "$D/selection.json" ] || { echo "$D/selection.json missing: run tools/run_v2.sh first (tile choice)"; exit 1; }
if ! done_ download; then
  say "downloading point clouds"
  python -m groundiff.data.download --out "$D/laz" --keys-file "$D/selection.json" --workers 8 \
    || python -m groundiff.data.download --out "$D/laz" --keys-file "$D/selection.json" --workers 4 \
    || echo "[warn] some tiles failed twice; carrying on without them"
  mark download
fi
if ! done_ ea_dtm; then
  say "downloading EA DTM rasters"
  python -m groundiff.data.ea_dtm --tiles "$D/laz" --out "$D/ea_dtm" \
    || python -m groundiff.data.ea_dtm --tiles "$D/laz" --out "$D/ea_dtm" \
    || echo "[warn] some DTM tiles failed twice; those scenes are skipped"
  mark ea_dtm
fi
if ! done_ preprocess; then
  say "rasterising"
  python -m groundiff.data.preprocess --points-dir "$D/laz" --dtm-dir "$D/ea_dtm" --out "$D/scenes" \
    --gsd 1.0 --workers "$WORKERS" --osm "$D/osm.json" --selection "$D/selection.json" \
    || echo "[warn] some scenes failed (listed above); training uses the rest"
  mark preprocess
fi
if [ ! -f "$D/split.json" ]; then
  say "train / val / test split"
  python -m groundiff.data.split --root "$D/scenes" --out "$D/split.json"
fi

say "training runs/v2_rdbm_range from scratch ($STEPS steps; watch: python -m groundiff.monitor runs/v2_rdbm_range --follow)"
python -m groundiff.train configs/v2_rdbm_range.json --set optim.total_steps="$STEPS" train.num_workers=6

say "test on the same $EVAL_SCENES v2 test scenes: RDBM, then Palette-style v2 (10-step and single-step), then the old model"
for m in runs/v2_rdbm_range runs/v2 runs/v2_1step runs/no_lastools_1step; do
  if [ -f "$m/best.pt" ]; then
    say "   $m"
    python -m groundiff.infer --checkpoint "$m/best.pt" --scenes "$D/scenes" --split-file "$D/split.json" \
      --split test --out "results/cmp_$(basename "$m")" --no-overlays --max-scenes "$EVAL_SCENES" --stride 256
  fi
done
say "finished"

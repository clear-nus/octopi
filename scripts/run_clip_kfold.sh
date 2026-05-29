#!/bin/bash
# K-fold CV on the current train_clip_config.yaml. Reports mean ± std of best
# val_mean across folds — use this as a final arbiter on a short list of configs
# from a cheap sweep, not as a sweep mechanism itself (K× the cost per config).
#
# Pipeline per fold:
#   1. _patch_constants.py rewrites TRAIN_OBJECTS / VAL_OBJECTS in src/utils/constants.py
#   2. process_dataset.py + generate_qa.py rebuild data/ for that split
#   3. train_clip.py trains
#   4. parse_clip_log.py extracts best val_mean
#
# Backups: src/utils/constants.py AND configs/train_clip_config.yaml are saved
# on entry and restored on exit (success, failure, or Ctrl-C).
#
# Usage:
#   bash scripts/run_clip_kfold.sh             # 5 folds, fold-build seed 0
#   K=3 FOLD_SEED=7 bash scripts/run_clip_kfold.sh

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

K="${K:-5}"
FOLD_SEED="${FOLD_SEED:-0}"
EXP_TAG="${EXP_TAG:-}"   # optional suffix to keep checkpoint dirs distinct across configs
DELETE_VIFICLIP="${DELETE_VIFICLIP:-0}"  # set to 1 to remove each fold's large vificlip.pt after parsing metrics
CONFIG="configs/train_clip_config.yaml"
CONSTANTS="src/utils/constants.py"
CONFIG_BAK="${CONFIG}.kfold_bak"
CONSTANTS_BAK="${CONSTANTS}.kfold_bak"
DATASET_PATH="dataset"
DATA_ROOT="data"
DATA_DIR="${DATA_ROOT}/kfold_${FOLD_SEED}${EXP_TAG:-base}"
SUMMARY="clip_kfold_summary.txt"

cp "$CONFIG" "$CONFIG_BAK"
cp "$CONSTANTS" "$CONSTANTS_BAK"
trap '
  echo "[kfold] restoring $CONFIG and $CONSTANTS"
  cp "$CONFIG_BAK" "$CONFIG"
  cp "$CONSTANTS_BAK" "$CONSTANTS"
  rm -f "$CONFIG_BAK" "$CONSTANTS_BAK"
' EXIT

: > "$SUMMARY"
echo "fold,val_mean,test_mean,test_combined" > "${SUMMARY}.csv"

# 1. Build folds.
python scripts/_build_kfold.py --k "$K" --seed "$FOLD_SEED"

latest_exp_dir() {
  local exp_id="$1"
  find exps -maxdepth 1 -type d -name "*_${exp_id}" | sort | tail -1
}

latest_log_field() {
  local exp_id="$1" field="$2" dir
  dir=$(latest_exp_dir "$exp_id")
  if [ -z "$dir" ] || [ ! -f "$dir/log.txt" ]; then
    echo "ERROR: no log for ${exp_id}" >&2
    return 1
  fi
  python src/utils/parse_clip_log.py "$dir/log.txt" --field "$field"
}

for ((i=0; i<K; i++)); do
  echo "================================================================"
  echo "[kfold] fold $i / $((K-1)) :: $(date)"
  echo "================================================================"

  # 2. Patch constants.py for this fold (always from the clean backup).
  cp "$CONSTANTS_BAK" "$CONSTANTS"
  python scripts/_patch_constants.py "scripts/kfold/fold_${i}.json"

  # 3. Regen data/ for this split.
  rm -rf "$DATA_DIR"
  mkdir -p "$DATA_ROOT"
  python src/utils/process_dataset.py --dataset_path "$DATASET_PATH" \
      --output_path "$DATA_DIR" --seed 0
  python src/utils/generate_qa.py --data_path "$DATA_DIR" --seed 0

  python src/utils/update_config.py --config_path "$CONFIG" \
      --key data_dir --value "$DATA_DIR" --key seed --value 0

  # 4. Train this fold.
  export EXP_ID="clip_kfold_${FOLD_SEED}${EXP_TAG}_f${i}"
  python src/train_clip.py

  v=$(latest_log_field "$EXP_ID" val_mean)
  tm=$(latest_log_field "$EXP_ID" test_mean)
  t=$(latest_log_field "$EXP_ID" test_combined)
  echo "[kfold] fold $i :: val_mean=$v test_mean=$tm test_combined=$t" | tee -a "$SUMMARY"
  echo "${i},${v},${tm},${t}" >> "${SUMMARY}.csv"
  if [ "$DELETE_VIFICLIP" = "1" ]; then
    exp_dir=$(latest_exp_dir "$EXP_ID")
    if [ -n "$exp_dir" ] && [ -f "$exp_dir/vificlip.pt" ]; then
      rm -f "$exp_dir/vificlip.pt"
      echo "[kfold] deleted $exp_dir/vificlip.pt"
    fi
  fi
done

echo "================================================================"
echo "[kfold] done. raw:"
cat "$SUMMARY"
echo "[kfold] aggregate:"
python - <<'PY'
import csv, statistics
rows = list(csv.DictReader(open("clip_kfold_summary.txt.csv")))
vs = [float(r["val_mean"]) for r in rows]
tms = [float(r["test_mean"]) for r in rows]
ts = [float(r["test_combined"]) for r in rows]
print(f"  val_mean      = {statistics.mean(vs):.3f} ± {statistics.stdev(vs):.3f}  (n={len(vs)})  [selection metric]")
print(f"  test_mean     = {statistics.mean(tms):.3f} ± {statistics.stdev(tms):.3f}  (n={len(tms)})  [per-property, report]")
print(f"  test_combined = {statistics.mean(ts):.3f} ± {statistics.stdev(ts):.3f}  (n={len(ts)})  [secondary]")
PY

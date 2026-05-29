#!/bin/bash
# Resume the interrupted paper-defaults + scaled CBS + decoupled-heads k-fold
# run for folds 3 and 4 only. Appends to the existing summary files.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

CONFIG="configs/train_clip_config.yaml"
CONSTANTS="src/utils/constants.py"
CONFIG_BAK="${CONFIG}.kfold_bak"
CONSTANTS_BAK="${CONSTANTS}.kfold_bak"
DATASET_PATH="dataset"
DATA_ROOT="data"
EXP_TAG="paper_cbs_decoupled_valmean"
DATA_DIR="${DATA_ROOT}/kfold_0${EXP_TAG}"
SUMMARY="clip_kfold_summary.txt"

set_cfg() {
  python src/utils/update_config.py --config_path "$CONFIG" "$@"
}

restore_defaults() {
  echo "[resume] restoring constants and plain baseline config"
  if [ -f "$CONSTANTS_BAK" ]; then
    cp "$CONSTANTS_BAK" "$CONSTANTS"
  fi
  set_cfg --key data_dir --value data \
          --key class_balanced_loss --value false \
          --key class_balance_mode --value scaled \
          --key decoupled_heads --value false \
          --key decoupled_head_dim --value 128 \
          --key swa --value false \
          --key swa_start_epoch --value 10 \
          --key ema_decay --value 0.0 \
          --key flip_p --value 0.5 \
          --key max_frames --value 8 \
          --key num_epochs --value 15 \
          --key weight_decay --value 0.05 \
          --key ranking_loss_weight --value 0.5 \
          --key ranking_margin --value 0.3 \
          --key batch_size --value 8 \
          --key gradient_accumulation_steps --value 4
}

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

trap restore_defaults EXIT

if [ ! -f "$CONSTANTS_BAK" ]; then
  echo "ERROR: missing $CONSTANTS_BAK; cannot safely resume from clean constants" >&2
  exit 1
fi

set_cfg --key class_balanced_loss --value true \
        --key class_balance_mode --value scaled \
        --key decoupled_heads --value true \
        --key decoupled_head_dim --value 128 \
        --key swa --value false \
        --key swa_start_epoch --value 10 \
        --key ema_decay --value 0.0 \
        --key flip_p --value 0.5 \
        --key max_frames --value 5 \
        --key num_epochs --value 30 \
        --key weight_decay --value 0.0 \
        --key ranking_loss_weight --value 0.0 \
        --key ranking_margin --value 0.3 \
        --key batch_size --value 8 \
        --key gradient_accumulation_steps --value 4

for i in 3 4; do
  echo "================================================================"
  echo "[resume] fold $i / 4 :: $(date)"
  echo "================================================================"

  cp "$CONSTANTS_BAK" "$CONSTANTS"
  python scripts/_patch_constants.py "scripts/kfold/fold_${i}.json"

  rm -rf "$DATA_DIR"
  mkdir -p "$DATA_ROOT"
  python src/utils/process_dataset.py --dataset_path "$DATASET_PATH" \
      --output_path "$DATA_DIR" --seed 0
  python src/utils/generate_qa.py --data_path "$DATA_DIR" --seed 0

  set_cfg --key data_dir --value "$DATA_DIR" --key seed --value 0

  export EXP_ID="clip_kfold_0${EXP_TAG}_f${i}"
  python src/train_clip.py

  v=$(latest_log_field "$EXP_ID" val_mean)
  tm=$(latest_log_field "$EXP_ID" test_mean)
  t=$(latest_log_field "$EXP_ID" test_combined)
  echo "[kfold] fold $i :: val_mean=$v test_mean=$tm test_combined=$t" | tee -a "$SUMMARY"
  echo "${i},${v},${tm},${t}" >> "${SUMMARY}.csv"

  exp_dir=$(latest_exp_dir "$EXP_ID")
  if [ -n "$exp_dir" ] && [ -f "$exp_dir/vificlip.pt" ]; then
    rm -f "$exp_dir/vificlip.pt"
    echo "[resume] deleted $exp_dir/vificlip.pt"
  fi
done

echo "================================================================"
echo "[resume] aggregate:"
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

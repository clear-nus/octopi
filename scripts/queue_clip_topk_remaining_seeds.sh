#!/bin/bash
# Remaining CLIP seed repeats for top-k EMA checkpoint averaging.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="configs/train_clip_config.yaml"
CONFIG_BAK="${CONFIG}.topk_remaining_bak"
DATASET_PATH="dataset"
DATA_ROOT="/tmp/octopi_clip_topk_remaining"
SUMMARY="clip_topk_remaining_summary.txt"
CSV="${SUMMARY}.csv"

cp "$CONFIG" "$CONFIG_BAK"
trap 'echo "[topk-remaining] restoring config"; cp "$CONFIG_BAK" "$CONFIG"; rm -f "$CONFIG_BAK"' EXIT

set_cfg() {
  python src/utils/update_config.py --config_path "$CONFIG" "$@"
}

latest_exp_dir() {
  local exp_id="$1"
  find exps -maxdepth 1 -type d -name "*_${exp_id}" | sort | tail -1
}

latest_log_field() {
  local exp_id="$1" selector="$2" field="$3" dir
  dir=$(latest_exp_dir "$exp_id")
  if [ -z "$dir" ] || [ ! -f "$dir/log.txt" ]; then
    echo "ERROR: no log for ${exp_id}" >&2
    return 1
  fi
  python src/utils/parse_clip_log.py "$dir/log.txt" --selector "$selector" --field "$field"
}

configure_variant() {
  local data_dir="$1" seed="$2"
  set_cfg --key data_dir --value "$data_dir" \
          --key cuda --value 7 \
          --key seed --value "$seed" \
          --key num_epochs --value 30 \
          --key max_frames --value 5 \
          --key ranking_loss_weight --value 0.0 \
          --key weight_decay --value 0.0 \
          --key label_smoothing --value 0.0 \
          --key lr --value 0.0003 \
          --key classifier_lr --value 0.0003 \
          --key class_balanced_loss --value true \
          --key class_balance_mode --value scaled \
          --key class_balance_strength --value 1.0 \
          --key class_balance_properties --value "[hardness,roughness,texture]" \
          --key decoupled_heads --value true \
          --key decoupled_head_dim --value 128 \
          --key ema_decay --value 0.98 \
          --key top_k_val_checkpoints --value 3 \
          --key swa --value false \
          --key flip_p --value 0.5 \
          --key rotation_degrees --value 0 \
          --key color_jitter --value 0.0 \
          --key gaussian_blur --value false
}

record_result() {
  local seed="$1" exp_id="$2"
  local bv btm btc tv ttm ttc
  bv=$(latest_log_field "$exp_id" val_mean val_mean)
  btm=$(latest_log_field "$exp_id" val_mean test_mean)
  btc=$(latest_log_field "$exp_id" val_mean test_combined)
  tv=$(latest_log_field "$exp_id" topk_avg val_mean)
  ttm=$(latest_log_field "$exp_id" topk_avg test_mean)
  ttc=$(latest_log_field "$exp_id" topk_avg test_combined)
  echo "[topk-remaining] seed=${seed} ${exp_id} :: best val_mean=${bv} test_mean=${btm} test_combined=${btc} :: topk val_mean=${tv} test_mean=${ttm} test_combined=${ttc}" | tee -a "$SUMMARY"
  echo "${seed},${exp_id},${bv},${btm},${btc},${tv},${ttm},${ttc}" >> "$CSV"
}

run_seed() {
  local seed="$1" exp_id="clip_seed_${seed}_repro_sorted_valmean_ema098_topk3"
  local data_dir="${DATA_ROOT}/seed_${seed}"

  echo "================================================================"
  echo "[topk-remaining] seed=${seed} :: $(date)"
  echo "================================================================"

  rm -rf "$data_dir"
  python src/utils/process_dataset.py --dataset_path "$DATASET_PATH" --output_path "$data_dir" --seed "$seed"
  python src/utils/generate_qa.py --data_path "$data_dir" --seed "$seed"

  configure_variant "$data_dir" "$seed"

  export EXP_ID="$exp_id"
  python src/train_clip.py
  record_result "$seed" "$exp_id"
}

aggregate_summary() {
  python - "$CSV" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print("[topk-remaining] aggregate:")
for key in ["best_test_mean", "best_test_combined", "topk_test_mean", "topk_test_combined"]:
    vals = [float(r[key]) for r in rows]
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  {key:19s} = {statistics.mean(vals):.3f} +/- {sd:.3f} (n={len(vals)})")
PY
}

mkdir -p "$DATA_ROOT"
: > "$SUMMARY"
echo "seed,exp_id,best_val_mean,best_test_mean,best_test_combined,topk_val_mean,topk_test_mean,topk_test_combined" > "$CSV"

for seed in 0 1 3 4; do
  run_seed "$seed"
done

aggregate_summary | tee -a "$SUMMARY"
echo "[topk-remaining] done at $(date)"

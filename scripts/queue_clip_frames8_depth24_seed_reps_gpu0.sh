#!/bin/bash
# Canonical-split CLIP seed repeats for the depth-24 VPT ablation.
# Matches the finalized 8-frame/no-top-k encoder config except:
#   prompt_depth_vision=24
# Runs seeds 0..4 on GPU 0.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="configs/train_clip_config.yaml"
CONFIG_BAK="${CONFIG}.frames8_depth24_gpu0_bak"
DATASET_PATH="dataset"
DATA_ROOT="/tmp/octopi_clip_frames8_depth24_gpu0"
SUMMARY="clip_seed_reps_frames8_depth24_ema098_gpu0.txt"
CSV="${SUMMARY}.csv"
CUDA_ID="${CUDA_ID:-0}"
SEEDS="${SEEDS:-0 1 2 3 4}"

cp "$CONFIG" "$CONFIG_BAK"
trap 'echo "[frames8-depth24-gpu0] restoring config"; cp "$CONFIG_BAK" "$CONFIG"; rm -f "$CONFIG_BAK"' EXIT

set_cfg() {
  python src/utils/update_config.py --config_path "$CONFIG" "$@"
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
  python src/utils/parse_clip_log.py "$dir/log.txt" --selector val_mean --field "$field"
}

configure_variant() {
  local data_dir="$1" seed="$2"
  set_cfg --key data_dir --value "$data_dir" \
          --key cuda --value "$CUDA_ID" \
          --key seed --value "$seed" \
          --key num_epochs --value 30 \
          --key max_frames --value 8 \
          --key num_context_vision --value 8 \
          --key prompt_depth_vision --value 24 \
          --key prompt_depth_text --value 12 \
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
          --key top_k_val_checkpoints --value 1 \
          --key swa --value false \
          --key flip_p --value 0.5 \
          --key rotation_degrees --value 0 \
          --key color_jitter --value 0.0 \
          --key gaussian_blur --value false
}

record_result() {
  local seed="$1" exp_id="$2"
  local v tm tc
  v=$(latest_log_field "$exp_id" val_mean)
  tm=$(latest_log_field "$exp_id" test_mean)
  tc=$(latest_log_field "$exp_id" test_combined)
  echo "[frames8-depth24-gpu0] seed=${seed} ${exp_id} :: val_mean=${v} test_mean=${tm} test_combined=${tc}" | tee -a "$SUMMARY"
  echo "${seed},${exp_id},${v},${tm},${tc}" >> "$CSV"
}

run_seed() {
  local seed="$1" exp_id="clip_seed_${seed}_repro_sorted_valmean_ema098_frames8_depth24_gpu0"
  local data_dir="${DATA_ROOT}/seed_${seed}"

  echo "================================================================"
  echo "[frames8-depth24-gpu0] seed=${seed} :: $(date)"
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
print("[frames8-depth24-gpu0] aggregate:")
for key in ["val_mean", "test_mean", "test_combined"]:
    vals = [float(r[key]) for r in rows]
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  {key:13s} = {statistics.mean(vals):.3f} +/- {sd:.3f} (n={len(vals)})")
PY
}

mkdir -p "$DATA_ROOT"
: > "$SUMMARY"
echo "seed,exp_id,val_mean,test_mean,test_combined" > "$CSV"

for seed in $SEEDS; do
  run_seed "$seed"
done

aggregate_summary | tee -a "$SUMMARY"
echo "[frames8-depth24-gpu0] done at $(date)"

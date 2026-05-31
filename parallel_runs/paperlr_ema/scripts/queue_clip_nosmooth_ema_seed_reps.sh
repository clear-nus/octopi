#!/bin/bash
# Canonical-split 5-seed repeat for the paper-close CLIP config with
# EMA enabled and no label smoothing. Runs in this isolated copy.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="configs/train_clip_config.yaml"
CONFIG_BAK="${CONFIG}.nosmooth_ema_seed_reps_bak"
DATASET_PATH="dataset"
DATA_DIR="data"
SUMMARY="clip_seed_reps_nosmooth_ema098.txt"
CSV="${SUMMARY}.csv"

cp "$CONFIG" "$CONFIG_BAK"
trap 'echo "[seed-reps] restoring config"; cp "$CONFIG_BAK" "$CONFIG"; rm -f "$CONFIG_BAK"' EXIT

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
  python src/utils/parse_clip_log.py "$dir/log.txt" --field "$field"
}

configure_variant() {
  set_cfg --key cuda --value 7 \
          --key data_dir --value "$DATA_DIR" \
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
  echo "[seed-reps] seed=${seed} ${exp_id} :: val_mean=${v} test_mean=${tm} test_combined=${tc}" | tee -a "$SUMMARY"
  echo "${seed},${exp_id},${v},${tm},${tc}" >> "$CSV"
}

run_seed() {
  local seed="$1" exp_id="clip_seed_${seed}_nosmooth_ema098"

  echo "================================================================"
  echo "[seed-reps] seed=${seed} :: $(date)"
  echo "================================================================"

  rm -rf "$DATA_DIR"
  python src/utils/process_dataset.py --dataset_path "$DATASET_PATH" --output_path "$DATA_DIR" --seed "$seed"
  python src/utils/generate_qa.py --data_path "$DATA_DIR" --seed "$seed"

  configure_variant
  set_cfg --key seed --value "$seed" --key data_dir --value "$DATA_DIR"

  export EXP_ID="$exp_id"
  python src/train_clip.py
  record_result "$seed" "$exp_id"
}

aggregate_summary() {
  python - "$CSV" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print("[seed-reps] aggregate:")
for key in ["val_mean", "test_mean", "test_combined"]:
    vals = [float(r[key]) for r in rows]
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  {key:13s} = {statistics.mean(vals):.3f} +/- {sd:.3f} (n={len(vals)})")
PY
}

: > "$SUMMARY"
echo "seed,exp_id,val_mean,test_mean,test_combined" > "$CSV"

for seed in 0 1 2 3 4; do
  run_seed "$seed"
done

aggregate_summary | tee -a "$SUMMARY"
echo "[seed-reps] done at $(date)"

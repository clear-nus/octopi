#!/bin/bash
# Single-seed CLIP run to test top-k EMA checkpoint averaging on the weak seed 2 case.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="configs/train_clip_config.yaml"
CONFIG_BAK="${CONFIG}.topk_seed2_bak"
DATASET_PATH="dataset"
DATA_DIR="/tmp/octopi_clip_topk_seed2"
SEED=2
EXP_ID_VALUE="clip_seed_2_repro_sorted_valmean_ema098_topk3"
SUMMARY="clip_topk_seed2_summary.txt"

cp "$CONFIG" "$CONFIG_BAK"
trap 'echo "[topk-seed2] restoring config"; cp "$CONFIG_BAK" "$CONFIG"; rm -f "$CONFIG_BAK"' EXIT

set_cfg() {
  python src/utils/update_config.py --config_path "$CONFIG" "$@"
}

latest_exp_dir() {
  find exps -maxdepth 1 -type d -name "*_${EXP_ID_VALUE}" | sort | tail -1
}

latest_log_field() {
  local selector="$1" field="$2" dir
  dir=$(latest_exp_dir)
  if [ -z "$dir" ] || [ ! -f "$dir/log.txt" ]; then
    echo "ERROR: no log for ${EXP_ID_VALUE}" >&2
    return 1
  fi
  python src/utils/parse_clip_log.py "$dir/log.txt" --selector "$selector" --field "$field"
}

echo "[topk-seed2] start $(date)" | tee "$SUMMARY"
rm -rf "$DATA_DIR"
python src/utils/process_dataset.py --dataset_path "$DATASET_PATH" --output_path "$DATA_DIR" --seed "$SEED"
python src/utils/generate_qa.py --data_path "$DATA_DIR" --seed "$SEED"

set_cfg --key data_dir --value "$DATA_DIR" \
        --key cuda --value 7 \
        --key seed --value "$SEED" \
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

export EXP_ID="$EXP_ID_VALUE"
python src/train_clip.py

DIR=$(latest_exp_dir)
{
  echo "[topk-seed2] exp_dir=${DIR}"
  echo "[topk-seed2] best_val_mean val_mean=$(latest_log_field val_mean val_mean) test_mean=$(latest_log_field val_mean test_mean) test_combined=$(latest_log_field val_mean test_combined)"
  echo "[topk-seed2] topk_avg val_mean=$(latest_log_field topk_avg val_mean) test_mean=$(latest_log_field topk_avg test_mean) test_combined=$(latest_log_field topk_avg test_combined)"
  echo "[topk-seed2] done $(date)"
} | tee -a "$SUMMARY"

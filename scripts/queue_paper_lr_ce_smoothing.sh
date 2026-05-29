#!/bin/bash
# K-fold paper-LR CE baselines:
#   1) pure CE (label_smoothing=0.0)
#   2) label-smoothed CE (label_smoothing=0.1)
# Deletes each fold's vificlip.pt after metrics are parsed to save space.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

CONFIG="configs/train_clip_config.yaml"

set_cfg() {
  python src/utils/update_config.py --config_path "$CONFIG" "$@"
}

restore_defaults() {
  echo "[queue] restoring plain baseline config"
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
          --key lr --value 0.0003 \
          --key classifier_lr --value 0.0003 \
          --key label_smoothing --value 0.1 \
          --key weight_decay --value 0.05 \
          --key ranking_loss_weight --value 0.5 \
          --key ranking_margin --value 0.3 \
          --key batch_size --value 8 \
          --key gradient_accumulation_steps --value 4
}

trap restore_defaults EXIT

run_ablation() {
  local tag="$1"
  local smoothing="$2"

  echo "================= ${tag} ================="
  set_cfg --key data_dir --value data \
          --key class_balanced_loss --value false \
          --key class_balance_mode --value scaled \
          --key decoupled_heads --value false \
          --key decoupled_head_dim --value 128 \
          --key swa --value false \
          --key swa_start_epoch --value 10 \
          --key ema_decay --value 0.0 \
          --key flip_p --value 0.5 \
          --key max_frames --value 5 \
          --key num_epochs --value 30 \
          --key lr --value 0.001 \
          --key classifier_lr --value 0.001 \
          --key label_smoothing --value "$smoothing" \
          --key weight_decay --value 0.0 \
          --key ranking_loss_weight --value 0.0 \
          --key ranking_margin --value 0.3 \
          --key batch_size --value 8 \
          --key gradient_accumulation_steps --value 4

  EXP_TAG="$tag" DELETE_VIFICLIP=1 bash scripts/run_clip_kfold.sh
  cp clip_kfold_summary.txt "clip_kfold_summary_${tag}.txt"
  cp clip_kfold_summary.txt.csv "clip_kfold_summary_${tag}.txt.csv"
}

echo "[queue] started paper LR CE/smoothing ablations at $(date)"
run_ablation paper_lr_pure_ce 0.0
run_ablation paper_lr_smooth01 0.1
echo "[queue] done at $(date)"

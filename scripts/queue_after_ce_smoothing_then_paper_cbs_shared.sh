#!/bin/bash
# Wait for the paper-LR CE-only and label-smoothed CE k-folds to finish, then
# run the next paper-close ablation: scaled class-balanced CE with shared heads.
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

complete_summary() {
  local file="$1"
  [[ -s "$file" ]] && [[ "$(wc -l < "$file")" -ge 6 ]]
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

echo "[queue] waiting for paper_lr_pure_ce and paper_lr_smooth01 at $(date)"
while ! complete_summary clip_kfold_summary_paper_lr_pure_ce.txt.csv || \
      ! complete_summary clip_kfold_summary_paper_lr_smooth01.txt.csv; do
  sleep 300
done

if complete_summary clip_kfold_summary_paper_lr_cbs_shared.txt.csv; then
  echo "[queue] paper_lr_cbs_shared summary already exists; skipping"
  exit 0
fi

echo "================= paper_lr_cbs_shared ================="
set_cfg --key data_dir --value data \
        --key class_balanced_loss --value true \
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
        --key label_smoothing --value 0.0 \
        --key weight_decay --value 0.0 \
        --key ranking_loss_weight --value 0.0 \
        --key ranking_margin --value 0.3 \
        --key batch_size --value 8 \
        --key gradient_accumulation_steps --value 4

EXP_TAG=paper_lr_cbs_shared DELETE_VIFICLIP=1 bash scripts/run_clip_kfold.sh
cp clip_kfold_summary.txt clip_kfold_summary_paper_lr_cbs_shared.txt
cp clip_kfold_summary.txt.csv clip_kfold_summary_paper_lr_cbs_shared.txt.csv

echo "[queue] done at $(date)"

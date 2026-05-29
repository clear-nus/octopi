#!/bin/bash
# K-fold run with paper-explicit CLIP defaults restored where practical, while
# keeping scaled class balancing and decoupled classifier heads. Selection remains
# val_mean because this is an ablation/selector run, not strict paper reproduction.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

CONFIG="configs/train_clip_config.yaml"

set_cfg() {
  python src/utils/update_config.py --config_path "$CONFIG" "$@"
}

restore_defaults() {
  echo "[queue] restoring plain baseline config"
  set_cfg --key class_balanced_loss --value false \
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

trap restore_defaults EXIT

echo "================= paper defaults + scaled + decoupled ================="
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

EXP_TAG=paper_cbs_decoupled_valmean DELETE_VIFICLIP=1 bash scripts/run_clip_kfold.sh
cp clip_kfold_summary.txt clip_kfold_summary_paper_cbs_decoupled_valmean.txt
cp clip_kfold_summary.txt.csv clip_kfold_summary_paper_cbs_decoupled_valmean.txt.csv

echo "[queue] done at $(date)"

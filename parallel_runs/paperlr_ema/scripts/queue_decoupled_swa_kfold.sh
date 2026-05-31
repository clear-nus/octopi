#!/bin/bash
# Runs one k-fold ablation with decoupled property heads and SWA tail averaging.
# The feature gates stay off in the checked-in config; this script enables them
# only for the run and restores the plain baseline afterward.

set -euo pipefail
cd /data/samson/octopi
source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

CONFIG="configs/train_clip_config.yaml"

set_cfg() { python src/utils/update_config.py --config_path "$CONFIG" "$@"; }

restore_defaults() {
  echo "[queue] restoring plain baseline config"
  set_cfg --key decoupled_heads --value false \
          --key decoupled_head_dim --value 128 \
          --key swa --value false \
          --key swa_start_epoch --value 10 \
          --key ema_decay --value 0.0 \
          --key class_balanced_loss --value false \
          --key flip_p --value 0.5
}

trap restore_defaults EXIT

echo "================= decoupled-heads + SWA k-fold ================="
set_cfg --key decoupled_heads --value true \
        --key decoupled_head_dim --value 128 \
        --key swa --value true \
        --key swa_start_epoch --value 10 \
        --key ema_decay --value 0.0 \
        --key class_balanced_loss --value true \
        --key flip_p --value 0.5

EXP_TAG=decoupled_swa_cbs bash scripts/run_clip_kfold.sh

echo "[queue] done at $(date)"

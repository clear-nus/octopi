#!/bin/bash
# K-fold test for whether EMA stabilizes the paper LR without label smoothing.
# Runs inside the isolated parallel_runs/paperlr_ema copy to avoid colliding
# with active jobs in the main repo.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="configs/train_clip_config.yaml"

python src/utils/update_config.py --config_path "$CONFIG" \
  --key cuda --value 7 \
  --key data_dir --value data \
  --key num_epochs --value 30 \
  --key max_frames --value 5 \
  --key ranking_loss_weight --value 0.0 \
  --key weight_decay --value 0.0 \
  --key label_smoothing --value 0.0 \
  --key lr --value 0.001 \
  --key classifier_lr --value 0.001 \
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

EXP_TAG="paperlr_nosmooth_cbs_dec_ema098" DELETE_VIFICLIP=1 K="${K:-5}" FOLD_SEED="${FOLD_SEED:-0}" bash scripts/run_clip_kfold.sh

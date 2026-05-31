#!/bin/bash
# One paper-close CLIP k-fold ablation:
#   paper_lr_cbs05_shared: paper LR/epochs, shared heads, CE only,
#                          scaled class-balanced loss at half strength
#                          applied to hardness, roughness, and texture.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

K="${K:-5}"
FOLD_SEED="${FOLD_SEED:-0}"
CONFIG="configs/train_clip_config.yaml"
CONSTANTS="src/utils/constants.py"
CONFIG_BAK="${CONFIG}.paper_lr_cbs05_bak"
CONSTANTS_BAK="${CONSTANTS}.paper_lr_cbs05_bak"

set_cfg() {
  python src/utils/update_config.py --config_path "$CONFIG" "$@"
}

restore_originals() {
  echo "[queue] restoring original config/constants"
  cp "$CONFIG_BAK" "$CONFIG"
  cp "$CONSTANTS_BAK" "$CONSTANTS"
  rm -f "$CONFIG_BAK" "$CONSTANTS_BAK"
}

cp "$CONFIG" "$CONFIG_BAK"
cp "$CONSTANTS" "$CONSTANTS_BAK"
trap restore_originals EXIT

set_cfg --key data_dir --value data \
        --key class_balanced_loss --value true \
        --key class_balance_mode --value scaled \
        --key class_balance_strength --value 0.5 \
        --key class_balance_properties --value "[hardness,roughness,texture]" \
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

EXP_TAG=paper_lr_cbs05_shared DELETE_VIFICLIP=1 K="$K" FOLD_SEED="$FOLD_SEED" bash scripts/run_clip_kfold.sh

cp clip_kfold_summary.txt "clip_kfold_summary_paper_lr_cbs05_shared.txt"
cp clip_kfold_summary.txt.csv "clip_kfold_summary_paper_lr_cbs05_shared.txt.csv"

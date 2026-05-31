#!/bin/bash
# Run three paper-close shared-head CLIP k-fold ablations at 15 epochs:
#   paper_lr15_cbs05_shared       : scaled CBS, half strength, no smoothing
#   paper_lr15_cbs05_smooth03     : scaled CBS, half strength, light smoothing
#   paper_lr15_cbs1_shared        : scaled CBS, full strength, no smoothing
#
# All variants apply class balancing to hardness, roughness, and texture.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

K="${K:-5}"
FOLD_SEED="${FOLD_SEED:-0}"
CONFIG="configs/train_clip_config.yaml"
CONSTANTS="src/utils/constants.py"
CONFIG_BAK="${CONFIG}.paper_lr15_cbs_bak"
CONSTANTS_BAK="${CONSTANTS}.paper_lr15_cbs_bak"

set_cfg() {
  python src/utils/update_config.py --config_path "$CONFIG" "$@"
}

restore_originals() {
  echo "[queue] restoring original config/constants"
  cp "$CONFIG_BAK" "$CONFIG"
  cp "$CONSTANTS_BAK" "$CONSTANTS"
  rm -f "$CONFIG_BAK" "$CONSTANTS_BAK"
}

configure_common() {
  set_cfg --key data_dir --value data \
          --key class_balanced_loss --value true \
          --key class_balance_mode --value scaled \
          --key class_balance_properties --value "[hardness,roughness,texture]" \
          --key decoupled_heads --value false \
          --key decoupled_head_dim --value 128 \
          --key swa --value false \
          --key swa_start_epoch --value 10 \
          --key ema_decay --value 0.0 \
          --key flip_p --value 0.5 \
          --key max_frames --value 5 \
          --key num_epochs --value 15 \
          --key lr --value 0.001 \
          --key classifier_lr --value 0.001 \
          --key weight_decay --value 0.0 \
          --key ranking_loss_weight --value 0.0 \
          --key ranking_margin --value 0.3 \
          --key batch_size --value 8 \
          --key gradient_accumulation_steps --value 4
}

run_variant() {
  local tag="$1"
  local strength="$2"
  local smoothing="$3"

  echo "================= ${tag} ================="
  configure_common
  set_cfg --key class_balance_strength --value "$strength" \
          --key label_smoothing --value "$smoothing"

  EXP_TAG="$tag" DELETE_VIFICLIP=1 K="$K" FOLD_SEED="$FOLD_SEED" bash scripts/run_clip_kfold.sh

  cp clip_kfold_summary.txt "clip_kfold_summary_${tag}.txt"
  cp clip_kfold_summary.txt.csv "clip_kfold_summary_${tag}.txt.csv"
}

cp "$CONFIG" "$CONFIG_BAK"
cp "$CONSTANTS" "$CONSTANTS_BAK"
trap restore_originals EXIT

run_variant paper_lr15_cbs05_shared 0.5 0.0
run_variant paper_lr15_cbs05_smooth03 0.5 0.03
run_variant paper_lr15_cbs1_shared 1.0 0.0

echo "[queue] done at $(date)"

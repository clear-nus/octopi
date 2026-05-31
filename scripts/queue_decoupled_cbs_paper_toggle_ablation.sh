#!/bin/bash
# Attribute the gap between:
#   paper_cbs_decoupled_valmean: 30 epochs, 5 frames, no ranking, no WD
#   cbs_decoupled_noswa:        15 epochs, 8 frames, ranking=0.5, WD=0.05
#
# Common settings are held fixed: scaled class-balanced CE, decoupled heads,
# no SWA/EMA, lr=classifier_lr=3e-4, label_smoothing=0.1.
# Each variant toggles one "newer" setting onto the paper-ish base.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

K="${K:-5}"
FOLD_SEED="${FOLD_SEED:-0}"
CONFIG="configs/train_clip_config.yaml"
CONSTANTS="src/utils/constants.py"
CONFIG_BAK="${CONFIG}.toggle_decoupled_cbs_bak"
CONSTANTS_BAK="${CONSTANTS}.toggle_decoupled_cbs_bak"

set_cfg() {
  python src/utils/update_config.py --config_path "$CONFIG" "$@"
}

restore_originals() {
  echo "[queue] restoring original config/constants"
  cp "$CONFIG_BAK" "$CONFIG"
  cp "$CONSTANTS_BAK" "$CONSTANTS"
  rm -f "$CONFIG_BAK" "$CONSTANTS_BAK"
}

configure_paperish_base() {
  set_cfg --key data_dir --value data \
          --key class_balanced_loss --value true \
          --key class_balance_mode --value scaled \
          --key class_balance_strength --value 1.0 \
          --key class_balance_properties --value "[hardness,roughness,texture]" \
          --key decoupled_heads --value true \
          --key decoupled_head_dim --value 128 \
          --key swa --value false \
          --key swa_start_epoch --value 10 \
          --key ema_decay --value 0.0 \
          --key flip_p --value 0.5 \
          --key max_frames --value 5 \
          --key num_epochs --value 30 \
          --key lr --value 0.0003 \
          --key classifier_lr --value 0.0003 \
          --key label_smoothing --value 0.1 \
          --key weight_decay --value 0.0 \
          --key ranking_loss_weight --value 0.0 \
          --key ranking_margin --value 0.3 \
          --key batch_size --value 8 \
          --key gradient_accumulation_steps --value 4
}

run_variant() {
  local tag="$1"
  shift

  echo "================= ${tag} ================="
  configure_paperish_base
  if [ "$#" -gt 0 ]; then
    set_cfg "$@"
  fi

  EXP_TAG="$tag" DELETE_VIFICLIP=1 K="$K" FOLD_SEED="$FOLD_SEED" bash scripts/run_clip_kfold.sh

  cp clip_kfold_summary.txt "clip_kfold_summary_${tag}.txt"
  cp clip_kfold_summary.txt.csv "clip_kfold_summary_${tag}.txt.csv"
}

cp "$CONFIG" "$CONFIG_BAK"
cp "$CONSTANTS" "$CONSTANTS_BAK"
trap restore_originals EXIT

run_variant paper_cbs_decoupled_plus_frames8 \
  --key max_frames --value 8

run_variant paper_cbs_decoupled_plus_epochs15 \
  --key num_epochs --value 15

run_variant paper_cbs_decoupled_plus_rank05 \
  --key ranking_loss_weight --value 0.5

run_variant paper_cbs_decoupled_plus_wd005 \
  --key weight_decay --value 0.05

echo "[queue] done at $(date)"

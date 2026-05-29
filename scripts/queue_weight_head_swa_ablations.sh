#!/bin/bash
# K-fold ablations for class-balance strength, shared/decoupled classifier heads,
# and SWA. Deletes each fold's vificlip.pt after metrics are parsed to save space.

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
          --key flip_p --value 0.5
}

trap restore_defaults EXIT

run_ablation() {
  local tag="$1"
  local balance_mode="$2"
  local decoupled="$3"
  local swa="$4"

  echo "================= ${tag} ================="
  set_cfg --key class_balanced_loss --value true \
          --key class_balance_mode --value "$balance_mode" \
          --key decoupled_heads --value "$decoupled" \
          --key decoupled_head_dim --value 128 \
          --key swa --value "$swa" \
          --key swa_start_epoch --value 10 \
          --key ema_decay --value 0.0 \
          --key flip_p --value 0.5

  EXP_TAG="$tag" DELETE_VIFICLIP=1 bash scripts/run_clip_kfold.sh
  cp clip_kfold_summary.txt "clip_kfold_summary_${tag}.txt"
  cp clip_kfold_summary.txt.csv "clip_kfold_summary_${tag}.txt.csv"
}

echo "[queue] started weight/head/SWA ablations at $(date)"

# Primary deconfounding runs for the already-completed endpoints:
#   full + shared + no-SWA       done previously as EXP_TAG=cb
#   scaled + decoupled + SWA     done previously as EXP_TAG=decoupled_swa_cbs
run_ablation cbs_shared_noswa scaled false false
run_ablation cbs_decoupled_noswa scaled true false
run_ablation cb_decoupled_noswa full true false

echo "[queue] done at $(date)"

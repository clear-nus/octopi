#!/bin/bash
# Two-phase CLIP encoder sweep.
#
# Phase A: rotation_degrees ∈ {0, 15, 20} at current (ranking_weight, ranking_margin) = (0.5, 1.0).
# Phase B: at the winning rotation, sweep four (ranking_weight, ranking_margin) combos.
#
# Each run reuses data/ (no regen) and the seed/data_dir already in the config.
# Winner is picked by best epoch's val mean per-property accuracy (matches train_clip.py's
# save criterion). Logs are streamed to clip_sweep_run.log via the launcher; per-run
# logs land in exps/<timestamp>_<exp_id>/log.txt as usual.
#
# Restores the original config on exit (success or failure).

set -euo pipefail

# Use the octopi conda env (base env's torch is built against a CUDA newer than the driver).
source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

CONFIG="configs/train_clip_config.yaml"
BACKUP="${CONFIG}.sweep_bak"
SUMMARY="clip_sweep_summary.txt"

cp "$CONFIG" "$BACKUP"
trap 'echo "[sweep] restoring $CONFIG from $BACKUP"; cp "$BACKUP" "$CONFIG"; rm -f "$BACKUP"' EXIT

: > "$SUMMARY"

update() {
  python src/utils/update_config.py --config_path "$CONFIG" "$@"
}

# Find the most recent exps/ dir for a given EXP_ID and print val_mean.
latest_log_field() {
  local exp_id="$1"
  local field="$2"
  local dir
  dir=$(find exps -maxdepth 1 -type d -name "*_${exp_id}" | sort | tail -1)
  if [ -z "$dir" ] || [ ! -f "$dir/log.txt" ]; then
    echo "ERROR: no log for ${exp_id}" >&2
    return 1
  fi
  python src/utils/parse_clip_log.py "$dir/log.txt" --field "$field"
}

run_one() {
  local tag="$1"
  export EXP_ID="clip_sweep_${tag}"
  echo "================================================================"
  echo "[sweep] $tag :: $(date)"
  echo "[sweep]   rotation_degrees=$(grep '^rotation_degrees:' $CONFIG)"
  echo "[sweep]   ranking_loss_weight=$(grep '^ranking_loss_weight:' $CONFIG)"
  echo "[sweep]   ranking_margin=$(grep '^ranking_margin:' $CONFIG)"
  echo "================================================================"
  python src/train_clip.py
  local val_mean test_combined
  val_mean=$(latest_log_field "$EXP_ID" val_mean)
  test_combined=$(latest_log_field "$EXP_ID" test_combined)
  echo "[sweep] $tag :: val_mean=$val_mean test_combined=$test_combined" | tee -a "$SUMMARY"
}

# --------- Phase A: rotation ---------
# Keep current ranking settings (0.5, 1.0).
update --key ranking_loss_weight --value 0.5 --key ranking_margin --value 1.0

best_rot=""
best_rot_val="-1"
for ROT in 0 15 20; do
  update --key rotation_degrees --value "$ROT"
  run_one "rot${ROT}"
  val=$(latest_log_field "clip_sweep_rot${ROT}" val_mean)
  if python -c "import sys; sys.exit(0 if float('$val') > float('$best_rot_val') else 1)"; then
    best_rot_val="$val"
    best_rot="$ROT"
  fi
done
echo "[sweep] Phase A winner: rotation_degrees=$best_rot (val_mean=$best_rot_val)" | tee -a "$SUMMARY"
update --key rotation_degrees --value "$best_rot"

# --------- Phase B: ranking (weight, margin) ---------
# Baseline (0.5, 1.0) already covered in Phase A; pick 4 new combos.
B_COMBOS=(
  "0.5 0.3"
  "0.5 0.5"
  "0.25 1.0"
  "1.0 1.0"
)
for combo in "${B_COMBOS[@]}"; do
  W=$(echo "$combo" | awk '{print $1}')
  M=$(echo "$combo" | awk '{print $2}')
  update --key ranking_loss_weight --value "$W" --key ranking_margin --value "$M"
  tag="rot${best_rot}_w${W}_m${M}"
  run_one "$tag"
done

echo "================================================================"
echo "[sweep] done. Summary in $SUMMARY"
cat "$SUMMARY"

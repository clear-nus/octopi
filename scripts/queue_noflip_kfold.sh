#!/bin/bash
# Waits for the in-progress EMA queue (PID passed as $1) to finish, then runs a
# single baseline k-fold with horizontal/vertical flip DISABLED (flip_p=0).
#
# Rationale: on GelSight the LED illumination directions are fixed, so a geometric
# flip breaks the geometry<->RGB (force-direction) consistency the sensor encodes.
# This ablates flip_p against the existing baseline k-fold (flip_p=0.5, test_mean
# 0.639). Everything else is plain baseline: ema_decay=0, class_balanced_loss=false.
#
# run_clip_kfold.sh backs up and restores the config itself, so we explicitly
# restore flip_p=0.5 afterward to leave the working config at the baseline default.

set -euo pipefail
cd /data/samson/octopi
source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

CONFIG="configs/train_clip_config.yaml"
WAIT_PID="${1:-}"

set_cfg() { python src/utils/update_config.py --config_path "$CONFIG" "$@"; }

if [ -n "$WAIT_PID" ]; then
  echo "[queue] waiting for EMA queue PID $WAIT_PID to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[queue] EMA queue finished; starting no-flip k-fold at $(date)"
  sleep 5
fi

echo "================= no-flip baseline k-fold ================="
set_cfg --key flip_p --value 0 --key ema_decay --value 0.0 --key class_balanced_loss --value false
EXP_TAG=noflip bash scripts/run_clip_kfold.sh

echo "[queue] restoring flip_p=0.5 baseline default"
set_cfg --key flip_p --value 0.5
echo "[queue] done at $(date)"

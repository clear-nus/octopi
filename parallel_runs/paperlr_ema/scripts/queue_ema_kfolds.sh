#!/bin/bash
# Waits for the in-progress cbs k-fold (PID passed as $1) to finish, then runs the
# two EMA k-folds needed to complete the 2x2 {EMA off/on} x {scaled weighting off/on}:
#   - EMA alone        (ema_decay=0.98, class_balanced_loss=false)  EXP_TAG=ema098
#   - EMA + scaled wts (ema_decay=0.98, class_balanced_loss=true)   EXP_TAG=emacbs098
# baseline (no EMA, no weighting) and cbs (scaled only) are already done.
#
# decay=0.98 chosen for the low-step regime (~135 optimizer steps total; half-life ~4 epochs).
# Restores config to plain baseline (ema 0, weighting off) at the end.

set -euo pipefail
cd /data/samson/octopi
source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

CONFIG="configs/train_clip_config.yaml"
WAIT_PID="${1:-}"

if [ -n "$WAIT_PID" ]; then
  echo "[queue] waiting for cbs run PID $WAIT_PID to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[queue] cbs run finished; starting EMA k-folds at $(date)"
  sleep 5
fi

set_cfg() { python src/utils/update_config.py --config_path "$CONFIG" "$@"; }

echo "================= EMA-alone k-fold ================="
set_cfg --key ema_decay --value 0.98 --key class_balanced_loss --value false
EXP_TAG=ema098 bash scripts/run_clip_kfold.sh

echo "================= EMA + scaled-weighting k-fold ================="
set_cfg --key ema_decay --value 0.98 --key class_balanced_loss --value true
EXP_TAG=emacbs098 bash scripts/run_clip_kfold.sh

echo "[queue] restoring plain baseline config (ema 0, weighting off)"
set_cfg --key ema_decay --value 0.0 --key class_balanced_loss --value false
echo "[queue] done at $(date)"

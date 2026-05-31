#!/bin/bash
# Wait for cuda:6 to become available, then run the endpoint ablation queue.

set -euo pipefail

GPU_ID="${GPU_ID:-6}"
MAX_USED_MB="${MAX_USED_MB:-8000}"
CHECK_SECONDS="${CHECK_SECONDS:-300}"
LOG_PATH="${LOG_PATH:-queue_decoupled_cbs_endpoint_ablation.log}"
VARIANTS="${VARIANTS:-base_paperish plus_frames8 plus_epochs15 plus_rank05 plus_wd005 target_newer}"
K="${K:-5}"
FOLD_SEED="${FOLD_SEED:-0}"

export VARIANTS K FOLD_SEED

echo "[wait] $(date) waiting for gpu ${GPU_ID} used memory <= ${MAX_USED_MB} MiB"
while true; do
  used_mb=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  echo "[wait] $(date) gpu ${GPU_ID} used=${used_mb} MiB"
  if [ "$used_mb" -le "$MAX_USED_MB" ]; then
    break
  fi
  sleep "$CHECK_SECONDS"
done

echo "[wait] $(date) gpu ${GPU_ID} available; starting endpoint ablation"
exec bash scripts/queue_decoupled_cbs_endpoint_ablation.sh

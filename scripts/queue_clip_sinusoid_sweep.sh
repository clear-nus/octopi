#!/bin/bash
# CLIP encoder seed sweep for the sinusoid comparison: 5f vs 8f.
# Our improved recipe (lr 3e-4, EMA 0.98, scaled CBS, decoupled heads, depth_vision=24)
# + the re-added per-frame sinusoidal PE (live in src/utils/model.py).
# Checkpoint SAVE + selection by val_combined (val_checkpoint_metric=val_combined).
# Reuses already-processed CLIP data (deterministic salient frames; seed only varies
# training randomness) so no reprocessing. max_frames applied at load.
#
# Params: FRAMES (5|8), CUDA, SEEDS ("0 1 ...").

set -euo pipefail
source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi
cd /data/samson/octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FRAMES="${FRAMES:-8}"
CUDA="${CUDA:-3}"
SEEDS="${SEEDS:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-15}"
CLIP_DATA="${CLIP_DATA:-/tmp/octopi_clip_frames8_depth24_gpu0/seed_0}"
SUMMARY="clip_sinusoid_f${FRAMES}_e${NUM_EPOCHS}_vc_gpu${CUDA}.csv"

set_cfg() { local cp="$1"; shift; python src/utils/update_config.py --config_path "$cp" "$@"; }
latest_exp_dir() { find exps -maxdepth 1 -type d -name "*_$1" | sort | tail -1; }

[ -f "$CLIP_DATA/train_samples.json" ] || { echo "Missing CLIP data $CLIP_DATA"; exit 1; }

for SEED in $SEEDS; do
  CFG="/tmp/octopi_clip_sinusoid_f${FRAMES}_s${SEED}.yaml"
  EXP_ID="clip_sinusoid_f${FRAMES}_s${SEED}_e${NUM_EPOCHS}_vc"
  echo "[sweep f${FRAMES} g${CUDA}] === seed ${SEED} :: $(date) ==="
  cp configs/train_clip_config.yaml "$CFG"
  set_cfg "$CFG" \
          --key data_dir --value "$CLIP_DATA" \
          --key cuda --value "$CUDA" \
          --key seed --value "$SEED" \
          --key num_epochs --value "$NUM_EPOCHS" \
          --key max_frames --value "$FRAMES" \
          --key num_context_vision --value 8 \
          --key prompt_depth_vision --value 24 \
          --key prompt_depth_text --value 12 \
          --key ranking_loss_weight --value 0.0 \
          --key weight_decay --value 0.0 \
          --key label_smoothing --value 0.0 \
          --key lr --value 0.0003 \
          --key classifier_lr --value 0.0003 \
          --key class_balanced_loss --value true \
          --key class_balance_mode --value scaled \
          --key class_balance_strength --value 1.0 \
          --key class_balance_properties --value "[hardness,roughness,texture]" \
          --key decoupled_heads --value true \
          --key decoupled_head_dim --value 128 \
          --key ema_decay --value 0.98 \
          --key top_k_val_checkpoints --value 1 \
          --key val_checkpoint_metric --value val_combined \
          --key swa --value false \
          --key flip_p --value 0.5 \
          --key rotation_degrees --value 0 \
          --key color_jitter --value 0.0 \
          --key gaussian_blur --value false
  EXP_ID="$EXP_ID" TRAIN_CLIP_CONFIG="$CFG" python src/train_clip.py
  rm -f "$CFG"
  DIR=$(latest_exp_dir "$EXP_ID")
  if [ -n "$DIR" ] && [ -f "$DIR/log.txt" ]; then
    VM=$(python src/utils/parse_clip_log.py "$DIR/log.txt" --selector val_combined --field val_mean)
    VC=$(python src/utils/parse_clip_log.py "$DIR/log.txt" --selector val_combined --field val_combined)
    TM=$(python src/utils/parse_clip_log.py "$DIR/log.txt" --selector val_combined --field test_mean)
    TC=$(python src/utils/parse_clip_log.py "$DIR/log.txt" --selector val_combined --field test_combined)
    echo "${FRAMES},${SEED},${VM},${VC},${TM},${TC},${DIR}" >> "$SUMMARY"
    echo "[sweep f${FRAMES} g${CUDA}] seed ${SEED}: val_combined=${VC} val_mean=${VM} test_mean=${TM} test_combined=${TC}"
  fi
done
echo "[sweep f${FRAMES} g${CUDA}] DONE $(date)"

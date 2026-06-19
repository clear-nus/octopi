#!/bin/bash
# Stage 1 + Stage 2 LLM flow using the seed-0 depth-24 CLIP encoder.
# Chained after the current GPU-5 LLM queue to avoid GPU collisions.
#
# Reuses prepared QA files and refuses to regenerate data.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

cd /data/samson/octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WAIT_SESSION="${WAIT_SESSION:-__none__}"
SEED="${SEED:-0}"
STAGE1_CUDA="${STAGE1_CUDA:-7}"
STAGE2_CUDA="${STAGE2_CUDA:-7}"
STAGE1_GPU_CONFIG="${STAGE1_GPU_CONFIG:-configs/gpu_config_7b_gpu7.json}"
STAGE2_GPU_CONFIG="${STAGE2_GPU_CONFIG:-configs/gpu_config_7b_gpu7.json}"
CONFIG_FILE="${CONFIG_FILE:-configs/train_llm_config.yaml}"
STAGE1_CONFIG="/tmp/octopi_stage1_depth24_${SEED}_${STAGE1_CUDA}.yaml"
STAGE2_CONFIG="/tmp/octopi_stage2_depth24_qkv_trainproj_${SEED}_${STAGE2_CUDA}.yaml"
DATA_DIR="${DATA_DIR:-/tmp/octopi_llm_frames8_encoder_full_stage12_data}"
ENCODER_PATH="${ENCODER_PATH:-exps/2026_06_07_15_08_20_train_clip_clip_seed_0_repro_sorted_valmean_ema098_frames8_depth24_gpu0/encoder.pt}"
RUN_TAG="${RUN_TAG:-depth24enc_gpu5chain}"

STAGE1_EXP_ID="fp0_d24_s1_full_noaux_8000_gpu7"
LORA_EXP_ID="fp0_d24_s2_frzemb_plr2e4_noaux_3000_gpu7"

echo "[llm-depth24-stage1-stage2] queued $(date)"
echo "[llm-depth24-stage1-stage2] waiting for tmux session $WAIT_SESSION"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  echo "[llm-depth24-stage1-stage2] waiting for $WAIT_SESSION ..."
  sleep 300
done

echo "[llm-depth24-stage1-stage2] start $(date)"
echo "[llm-depth24-stage1-stage2] seed=${SEED}"
echo "[llm-depth24-stage1-stage2] stage1_cuda=${STAGE1_CUDA} stage2_cuda=${STAGE2_CUDA}"
echo "[llm-depth24-stage1-stage2] encoder=${ENCODER_PATH}"
echo "[llm-depth24-stage1-stage2] data=${DATA_DIR}"
echo "[llm-depth24-stage1-stage2] run_tag=${RUN_TAG}"

cleanup() {
  echo "[llm-depth24-stage1-stage2] removing private configs"
  rm -f "$STAGE1_CONFIG" "$STAGE2_CONFIG"
}
trap cleanup EXIT

set_cfg() {
  local config_path="$1"
  shift
  python src/utils/update_config.py --config_path "$config_path" "$@"
}

latest_exp_dir() {
  local exp_id="$1"
  find exps -maxdepth 1 -type d -name "*_${exp_id}" | sort | tail -1
}

if [ ! -f "$ENCODER_PATH" ]; then
  echo "Missing depth-24 encoder checkpoint: $ENCODER_PATH"
  exit 1
fi

if [ ! -f "$DATA_DIR/train_qa.json" ] || [ ! -f "$DATA_DIR/val_qa.json" ] || [ ! -f "$DATA_DIR/test_qa.json" ] || [ ! -f "$DATA_DIR/val_opd_qa.json" ] || [ ! -f "$DATA_DIR/test_opd_qa.json" ]; then
  echo "Missing existing QA files in $DATA_DIR; refusing to process/regenerate data in this script."
  exit 1
fi

echo "[llm-depth24-stage1-stage2] Stage 1: full answer loss, no LoRA, train-only, 8000 steps"
cp "$CONFIG_FILE" "$STAGE1_CONFIG"
set_cfg "$STAGE1_CONFIG" \
        --key data_dir --value "$DATA_DIR" \
        --key train_files --value "[$DATA_DIR/train_qa.json]" \
        --key val_files --value "[$DATA_DIR/val_opd_qa.json, $DATA_DIR/val_qa.json]" \
        --key test_files --value "[$DATA_DIR/test_opd_qa.json, $DATA_DIR/test_qa.json]" \
        --key seed --value "$SEED" \
        --key cuda --value "$STAGE1_CUDA" \
        --key gpu_config --value "$STAGE1_GPU_CONFIG" \
        --key encoder_path --value "$ENCODER_PATH" \
        --key max_frames --value 8 \
        --key num_context_vision --value 8 \
        --key prompt_depth_vision --value 24 \
        --key use_lora --value False \
        --key train_all_token_embeddings --value True \
        --key lora_trained --value False \
        --key max_train_steps --value 8000 \
        --key val_freq --value null \
        --key val_checkpoint_metric --value loss \
        --key projection_path --value null \
        --key tokenizer_path --value null \
        --key llm_path --value null \
        --key exps_path --value exps \
        --key freeze_encoder --value True \
        --key freeze_projection --value False \
        --key modules_to_save --value "[embed_tokens]" \
        --key projection_lr --value 0.00002 \
        --key llm_lr --value 0.00002 \
        --key warmup_steps --value 20 \
        --key llm_gradient_accumulation_steps --value 16 \
        --key task_balanced_training --value False \
        --key conclusion_only_loss --value false \
        --key conclusion_loss_weight --value 0.0 \
        --key conclusion_loss_task_groups --value null \
        --key pom_conclusion_loss_weight --value null \
        --key pom_option_id_format --value false \
        --key pom_candidate_permutation_augmentation --value 0 \
        --key multi_object_slot_permutation_augmentation --value 0 \
        --key multi_object_label_order_permutation_augmentation --value 0 \
        --key opd_consistency_loss_weight --value 0.0 \
        --key opd_consistency_lr --value 0.0001 \
        --key train --value True \
        --key val --value False \
        --key test --value False \
        --key tta_passes --value 1

export EXP_ID="$STAGE1_EXP_ID"
TRAIN_LLM_CONFIG="$STAGE1_CONFIG" python src/train_llm.py

STAGE1_DIR=$(latest_exp_dir "$STAGE1_EXP_ID")
if [ -z "$STAGE1_DIR" ] || [ ! -f "$STAGE1_DIR/final_project.pt" ] || [ ! -f "$STAGE1_DIR/final_llm_weights.pt" ] || [ ! -d "$STAGE1_DIR/tokenizer" ]; then
  echo "Stage 1 final output missing required artifacts: ${STAGE1_DIR:-<none>}"
  exit 1
fi
echo "[llm-depth24-stage1-stage2] Stage 1 final output: $STAGE1_DIR"

echo "[llm-depth24-stage1-stage2] Stage 2: q/k/v LoRA from depth-24 Stage 1 final, train projector, final-only test"
cp "$CONFIG_FILE" "$STAGE2_CONFIG"
set_cfg "$STAGE2_CONFIG" \
        --key data_dir --value "$DATA_DIR" \
        --key train_files --value "[$DATA_DIR/train_qa.json]" \
        --key val_files --value "[$DATA_DIR/val_opd_qa.json, $DATA_DIR/val_qa.json]" \
        --key test_files --value "[$DATA_DIR/test_opd_qa.json, $DATA_DIR/test_qa.json]" \
        --key seed --value "$SEED" \
        --key cuda --value "$STAGE2_CUDA" \
        --key gpu_config --value "$STAGE2_GPU_CONFIG" \
        --key encoder_path --value "$ENCODER_PATH" \
        --key max_frames --value 8 \
        --key num_context_vision --value 8 \
        --key prompt_depth_vision --value 24 \
        --key use_lora --value True \
        --key lora_trained --value False \
        --key max_train_steps --value 3000 \
        --key projection_path --value "$STAGE1_DIR/final_project.pt" \
        --key tokenizer_path --value "$STAGE1_DIR/tokenizer" \
        --key llm_path --value "$STAGE1_DIR/final_llm_weights.pt" \
        --key exps_path --value exps \
        --key freeze_encoder --value True \
        --key freeze_projection --value False \
        --key modules_to_save --value "[embed_tokens]" \
        --key projection_lr --value 0.00002 \
        --key llm_lr --value 0.0001 \
        --key projection_lr --value 0.0002 \
        --key freeze_new_token_embeddings --value True \
        --key warmup_steps --value 50 \
        --key val_freq --value null \
        --key val_checkpoint_metric --value loss \
        --key r --value 128 \
        --key lora_alpha --value 256 \
        --key lora_dropout --value 0.05 \
        --key target_modules --value "[q_proj, k_proj, v_proj]" \
        --key llm_gradient_accumulation_steps --value 16 \
        --key task_balanced_training --value False \
        --key conclusion_only_loss --value false \
        --key conclusion_loss_weight --value 0.0 \
        --key conclusion_loss_task_groups --value null \
        --key pom_conclusion_loss_weight --value null \
        --key pom_option_id_format --value false \
        --key pom_candidate_permutation_augmentation --value 0 \
        --key multi_object_slot_permutation_augmentation --value 0 \
        --key multi_object_label_order_permutation_augmentation --value 0 \
        --key opd_consistency_loss_weight --value 0.0 \
        --key opd_consistency_lr --value 0.0001 \
        --key train --value True \
        --key val --value False \
        --key test --value True \
        --key tta_passes --value 1

export EXP_ID="$LORA_EXP_ID"
TRAIN_LLM_CONFIG="$STAGE2_CONFIG" python src/train_llm.py

LORA_DIR=$(latest_exp_dir "$LORA_EXP_ID")
echo "[llm-depth24-stage1-stage2] Stage 2 output: ${LORA_DIR:-<none>}"
echo "[llm-depth24-stage1-stage2] done $(date)"

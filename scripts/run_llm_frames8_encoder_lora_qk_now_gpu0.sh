#!/bin/bash
# Immediate paper-aligned Stage 2 LoRA run on GPU 0.
# Uses the completed frames8 Stage 1 checkpoint and only q_proj/k_proj LoRA.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=0
CUDA=0
CONFIG_FILE="configs/train_llm_config.yaml"
CONFIG_BAK="${CONFIG_FILE}.frames8_lora_qk_now_gpu0_bak"
DATA_DIR="/tmp/octopi_llm_frames8_encoder_full_stage12_data"
ENCODER_PATH="exps/2026_06_01_15_28_38_train_clip_clip_seed_0_repro_sorted_valmean_ema098_frames8_notopk/encoder.pt"
STAGE1_DIR="exps/2026_06_01_19_11_15_train_llm_train_val_test_vicuna-7b_3200_full_pipeline_0_frames8enc_fullloss_stage1"
LORA_EXP_ID="full_pipeline_${SEED}_frames8enc_fullloss_lora_qk_r128_a256_lr0.0002_proj2e-5_gpu${CUDA}"

cp "$CONFIG_FILE" "$CONFIG_BAK"
trap 'echo "[llm-qk-now] restoring config"; cp "$CONFIG_BAK" "$CONFIG_FILE"; rm -f "$CONFIG_BAK"' EXIT

set_cfg() {
  python src/utils/update_config.py --config_path "$CONFIG_FILE" "$@"
}

latest_exp_dir() {
  local exp_id="$1"
  find exps -maxdepth 1 -type d -name "*_${exp_id}" | sort | tail -1
}

echo "[llm-qk-now] start $(date)"
echo "[llm-qk-now] cuda=${CUDA}"

if [ ! -f "$ENCODER_PATH" ]; then
  echo "Missing encoder checkpoint: $ENCODER_PATH"
  exit 1
fi
if [ ! -f "$STAGE1_DIR/best_project.pt" ] || [ ! -f "$STAGE1_DIR/best_llm_weights.pt" ] || [ ! -d "$STAGE1_DIR/tokenizer" ]; then
  echo "Stage 1 output missing required artifacts: $STAGE1_DIR"
  exit 1
fi

if [ ! -f "$DATA_DIR/train_qa.json" ] || [ ! -f "$DATA_DIR/val_qa.json" ] || [ ! -f "$DATA_DIR/test_qa.json" ]; then
  echo "Missing existing QA files in $DATA_DIR; refusing to process/regenerate data in this script."
  exit 1
fi

echo "[llm-qk-now] Stage 2: full answer loss, LoRA target_modules=[q_proj,k_proj], r=128 alpha=256"
set_cfg --key data_dir --value "$DATA_DIR" \
        --key seed --value "$SEED" \
        --key cuda --value "$CUDA" \
        --key gpu_config --value configs/gpu_config_7b_gpu0.json \
        --key encoder_path --value "$ENCODER_PATH" \
        --key max_frames --value 8 \
        --key use_lora --value True \
        --key lora_trained --value False \
        --key max_train_steps --value 3000 \
        --key projection_path --value "$STAGE1_DIR/best_project.pt" \
        --key tokenizer_path --value "$STAGE1_DIR/tokenizer" \
        --key llm_path --value "$STAGE1_DIR/best_llm_weights.pt" \
        --key exps_path --value exps \
        --key freeze_projection --value False \
        --key modules_to_save --value "[embed_tokens]" \
        --key projection_lr --value 0.00002 \
        --key llm_lr --value 0.0002 \
        --key warmup_steps --value 20 \
        --key val_freq --value 150 \
        --key r --value 128 \
        --key lora_alpha --value 256 \
        --key lora_dropout --value 0.05 \
        --key target_modules --value "[q_proj, k_proj]" \
        --key llm_gradient_accumulation_steps --value 16 \
        --key conclusion_only_loss --value false \
        --key train --value True \
        --key val --value True \
        --key test --value True \
        --key tta_passes --value 1

export EXP_ID="$LORA_EXP_ID"
python src/train_llm.py

echo "[llm-qk-now] completed LoRA: $(latest_exp_dir "$LORA_EXP_ID")"
echo "[llm-qk-now] done $(date)"

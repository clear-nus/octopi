#!/bin/bash
# Paper-aligned LLM rerun using the accepted 8-frame encoder:
# Stage 1 full-loss alignment for 8000 samples, then q_proj/k_proj LoRA.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=0
CUDA=6
CONFIG_FILE="configs/train_llm_config.yaml"
CONFIG_BAK="${CONFIG_FILE}.frames8_stage1_8000_qk_bak"
DATA_DIR="/tmp/octopi_llm_frames8_encoder_full_stage12_data"
ENCODER_PATH="exps/2026_06_01_15_28_38_train_clip_clip_seed_0_repro_sorted_valmean_ema098_frames8_notopk/encoder.pt"
STAGE1_EXP_ID="full_pipeline_${SEED}_frames8enc_fullloss_stage1_8000_paperaligned"
LORA_EXP_ID="full_pipeline_${SEED}_frames8enc_fullloss_stage1_8000_lora_qk_r128_a256_lr0.0002_proj2e-5"

cp "$CONFIG_FILE" "$CONFIG_BAK"
trap 'echo "[llm-stage1-8000-qk] restoring config"; cp "$CONFIG_BAK" "$CONFIG_FILE"; rm -f "$CONFIG_BAK"' EXIT

set_cfg() {
  python src/utils/update_config.py --config_path "$CONFIG_FILE" "$@"
}

latest_exp_dir() {
  local exp_id="$1"
  find exps -maxdepth 1 -type d -name "*_${exp_id}" | sort | tail -1
}

echo "[llm-stage1-8000-qk] start $(date)"
echo "[llm-stage1-8000-qk] cuda=${CUDA}"
echo "[llm-stage1-8000-qk] encoder=${ENCODER_PATH}"
echo "[llm-stage1-8000-qk] data=${DATA_DIR}"

if [ ! -f "$ENCODER_PATH" ]; then
  echo "Missing encoder checkpoint: $ENCODER_PATH"
  exit 1
fi

if [ ! -f "$DATA_DIR/train_qa.json" ] || [ ! -f "$DATA_DIR/val_qa.json" ] || [ ! -f "$DATA_DIR/test_qa.json" ] || [ ! -f "$DATA_DIR/val_opd_qa.json" ] || [ ! -f "$DATA_DIR/test_opd_qa.json" ]; then
  echo "Missing existing QA files in $DATA_DIR; refusing to process/regenerate data in this script."
  exit 1
fi

echo "[llm-stage1-8000-qk] Stage 1: full answer loss, no LoRA, 8000 samples"
set_cfg --key data_dir --value "$DATA_DIR" \
        --key train_files --value "[$DATA_DIR/train_qa.json]" \
        --key val_files --value "[$DATA_DIR/val_opd_qa.json, $DATA_DIR/val_qa.json]" \
        --key test_files --value "[$DATA_DIR/test_opd_qa.json, $DATA_DIR/test_qa.json]" \
        --key seed --value "$SEED" \
        --key cuda --value "$CUDA" \
        --key gpu_config --value configs/gpu_config_7b.json \
        --key encoder_path --value "$ENCODER_PATH" \
        --key max_frames --value 8 \
        --key use_lora --value False \
        --key lora_trained --value False \
        --key max_train_steps --value 8000 \
        --key val_freq --value 400 \
        --key projection_path --value null \
        --key tokenizer_path --value null \
        --key llm_path --value null \
        --key exps_path --value exps \
        --key freeze_encoder --value True \
        --key freeze_projection --value False \
        --key modules_to_save --value "[embed_tokens]" \
        --key projection_lr --value 0.00005 \
        --key llm_lr --value 0.00005 \
        --key warmup_steps --value 20 \
        --key llm_gradient_accumulation_steps --value 16 \
        --key conclusion_only_loss --value false \
        --key train --value True \
        --key val --value True \
        --key test --value True \
        --key tta_passes --value 1

export EXP_ID="$STAGE1_EXP_ID"
python src/train_llm.py

STAGE1_DIR=$(latest_exp_dir "$STAGE1_EXP_ID")
if [ -z "$STAGE1_DIR" ] || [ ! -f "$STAGE1_DIR/best_project.pt" ] || [ ! -f "$STAGE1_DIR/best_llm_weights.pt" ] || [ ! -d "$STAGE1_DIR/tokenizer" ]; then
  echo "Stage 1 output missing required artifacts: ${STAGE1_DIR:-<none>}"
  exit 1
fi
echo "[llm-stage1-8000-qk] Stage 1 output: $STAGE1_DIR"

echo "[llm-stage1-8000-qk] Stage 2: q_proj/k_proj LoRA, r=128 alpha=256, lr=2e-4, proj_lr=2e-5"
set_cfg --key data_dir --value "$DATA_DIR" \
        --key train_files --value "[$DATA_DIR/train_qa.json]" \
        --key val_files --value "[$DATA_DIR/val_opd_qa.json, $DATA_DIR/val_qa.json]" \
        --key test_files --value "[$DATA_DIR/test_opd_qa.json, $DATA_DIR/test_qa.json]" \
        --key seed --value "$SEED" \
        --key cuda --value "$CUDA" \
        --key gpu_config --value configs/gpu_config_7b.json \
        --key encoder_path --value "$ENCODER_PATH" \
        --key max_frames --value 8 \
        --key use_lora --value True \
        --key lora_trained --value False \
        --key max_train_steps --value 3000 \
        --key projection_path --value "$STAGE1_DIR/best_project.pt" \
        --key tokenizer_path --value "$STAGE1_DIR/tokenizer" \
        --key llm_path --value "$STAGE1_DIR/best_llm_weights.pt" \
        --key exps_path --value exps \
        --key freeze_encoder --value True \
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

echo "[llm-stage1-8000-qk] completed LoRA: $(latest_exp_dir "$LORA_EXP_ID")"
echo "[llm-stage1-8000-qk] done $(date)"

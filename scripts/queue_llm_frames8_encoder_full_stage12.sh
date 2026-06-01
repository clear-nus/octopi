#!/bin/bash
# Full-loss LLM Stage 1 + Stage 2 using the 8-frame seed-0 CLIP encoder.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=0
CONFIG_FILE="configs/train_llm_config.yaml"
CONFIG_BAK="${CONFIG_FILE}.frames8_encoder_full_stage12_bak"
DATASET_PATH="dataset"
DATA_DIR="/tmp/octopi_llm_frames8_encoder_full_stage12_data"
ENCODER_PATH="exps/2026_06_01_15_28_38_train_clip_clip_seed_0_repro_sorted_valmean_ema098_frames8_notopk/encoder.pt"
STAGE1_EXP_ID="full_pipeline_${SEED}_frames8enc_fullloss_stage1"
LORA_EXP_ID="full_pipeline_${SEED}_frames8enc_fullloss_lora_r128_lr0.0002"

cp "$CONFIG_FILE" "$CONFIG_BAK"
trap 'echo "[llm-frames8enc] restoring config"; cp "$CONFIG_BAK" "$CONFIG_FILE"; rm -f "$CONFIG_BAK"' EXIT

set_cfg() {
  python src/utils/update_config.py --config_path "$CONFIG_FILE" "$@"
}

latest_exp_dir() {
  local exp_id="$1"
  find exps -maxdepth 1 -type d -name "*_${exp_id}" | sort | tail -1
}

if [ ! -f "$ENCODER_PATH" ]; then
  echo "Missing encoder checkpoint: $ENCODER_PATH"
  exit 1
fi

echo "[llm-frames8enc] start $(date)"
echo "[llm-frames8enc] encoder: $ENCODER_PATH"
echo "[llm-frames8enc] regenerating data for seed ${SEED} in ${DATA_DIR}"
rm -rf "$DATA_DIR"
python src/utils/process_dataset.py --dataset_path "$DATASET_PATH" --output_path "$DATA_DIR" --seed "$SEED"
python src/utils/generate_qa.py --data_path "$DATA_DIR" --seed "$SEED"

echo "[llm-frames8enc] Stage 1: full answer loss, no LoRA, max_frames=8"
set_cfg --key data_dir --value "$DATA_DIR" \
        --key seed --value "$SEED" \
        --key cuda --value 6 \
        --key encoder_path --value "$ENCODER_PATH" \
        --key max_frames --value 8 \
        --key use_lora --value False \
        --key max_train_steps --value 3200 \
        --key val_freq --value 400 \
        --key projection_path --value null \
        --key tokenizer_path --value null \
        --key llm_path --value null \
        --key exps_path --value exps \
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
echo "[llm-frames8enc] Stage 1 output: $STAGE1_DIR"

echo "[llm-frames8enc] Stage 2: full answer loss, LoRA r=128 lr=2e-4 warmup=20"
set_cfg --key data_dir --value "$DATA_DIR" \
        --key seed --value "$SEED" \
        --key cuda --value 6 \
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
        --key projection_lr --value 0.0002 \
        --key llm_lr --value 0.0002 \
        --key warmup_steps --value 20 \
        --key val_freq --value 150 \
        --key r --value 128 \
        --key lora_alpha --value 128 \
        --key lora_dropout --value 0.05 \
        --key target_modules --value "[q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]" \
        --key llm_gradient_accumulation_steps --value 16 \
        --key conclusion_only_loss --value false \
        --key train --value True \
        --key val --value True \
        --key test --value True \
        --key tta_passes --value 1

export EXP_ID="$LORA_EXP_ID"
python src/train_llm.py

echo "[llm-frames8enc] completed LoRA: $(latest_exp_dir "$LORA_EXP_ID")"
echo "[llm-frames8enc] done $(date)"

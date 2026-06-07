#!/bin/bash
# Full LLM flow:
#   1. Stage 1 tactile-language alignment, train-only, save final checkpoint.
#   2. Stage 2 q/k/v LoRA from that Stage 1 final checkpoint, final-only test.
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

SEED="${SEED:-0}"
STAGE1_CUDA="${STAGE1_CUDA:-4}"
STAGE2_CUDA="${STAGE2_CUDA:-5}"
STAGE1_GPU_CONFIG="${STAGE1_GPU_CONFIG:-configs/gpu_config_7b_gpu4.json}"
STAGE2_GPU_CONFIG="${STAGE2_GPU_CONFIG:-configs/gpu_config_7b_gpu5.json}"
CONFIG_FILE="${CONFIG_FILE:-configs/train_llm_config.yaml}"
STAGE1_CONFIG="/tmp/octopi_stage1_final_${SEED}_${STAGE1_CUDA}.yaml"
STAGE2_CONFIG="/tmp/octopi_stage2_qkv_${SEED}_${STAGE2_CUDA}.yaml"
DATA_DIR="${DATA_DIR:-/tmp/octopi_llm_frames8_encoder_full_stage12_data}"
ENCODER_PATH="${ENCODER_PATH:-exps/2026_06_01_15_28_38_train_clip_clip_seed_0_repro_sorted_valmean_ema098_frames8_notopk/encoder.pt}"
RUN_TAG="${RUN_TAG:-auto}"

STAGE1_EXP_ID="full_pipeline_${SEED}_frames8enc_fullloss_stage1_8000_finalonly_lr2e-5_${RUN_TAG}"
LORA_EXP_ID="full_pipeline_${SEED}_frames8enc_fullloss_stage1_8000final_lora_qkv_r128_a256_lr0.0001_freezeproj_finalonly_originalpom_reasonconcl0.25_pom0.5_cons0.02_3000_${RUN_TAG}"

echo "[llm-stage1-stage2-auto] start $(date)"
echo "[llm-stage1-stage2-auto] seed=${SEED}"
echo "[llm-stage1-stage2-auto] stage1_cuda=${STAGE1_CUDA} stage2_cuda=${STAGE2_CUDA}"
echo "[llm-stage1-stage2-auto] encoder=${ENCODER_PATH}"
echo "[llm-stage1-stage2-auto] data=${DATA_DIR}"
echo "[llm-stage1-stage2-auto] run_tag=${RUN_TAG}"

cleanup() {
  echo "[llm-stage1-stage2-auto] removing private configs"
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
  echo "Missing encoder checkpoint: $ENCODER_PATH"
  exit 1
fi

if [ ! -f "$DATA_DIR/train_qa.json" ] || [ ! -f "$DATA_DIR/val_qa.json" ] || [ ! -f "$DATA_DIR/test_qa.json" ] || [ ! -f "$DATA_DIR/val_opd_qa.json" ] || [ ! -f "$DATA_DIR/test_opd_qa.json" ]; then
  echo "Missing existing QA files in $DATA_DIR; refusing to process/regenerate data in this script."
  exit 1
fi

echo "[llm-stage1-stage2-auto] Stage 1: full answer loss, no LoRA, train-only, 8000 steps"
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
        --key use_lora --value False \
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
echo "[llm-stage1-stage2-auto] Stage 1 final output: $STAGE1_DIR"

echo "[llm-stage1-stage2-auto] Stage 2: q/k/v LoRA from Stage 1 final, final-only test"
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
        --key use_lora --value True \
        --key lora_trained --value False \
        --key max_train_steps --value 3000 \
        --key projection_path --value "$STAGE1_DIR/final_project.pt" \
        --key tokenizer_path --value "$STAGE1_DIR/tokenizer" \
        --key llm_path --value "$STAGE1_DIR/final_llm_weights.pt" \
        --key exps_path --value exps \
        --key freeze_encoder --value True \
        --key freeze_projection --value True \
        --key modules_to_save --value "[embed_tokens]" \
        --key projection_lr --value 0.00002 \
        --key llm_lr --value 0.0001 \
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
        --key conclusion_loss_weight --value 0.25 \
        --key conclusion_loss_task_groups --value "[pom, pc, pss]" \
        --key pom_conclusion_loss_weight --value 0.5 \
        --key pom_option_id_format --value false \
        --key pom_candidate_permutation_augmentation --value 0 \
        --key multi_object_slot_permutation_augmentation --value 0 \
        --key multi_object_label_order_permutation_augmentation --value 0 \
        --key opd_consistency_loss_weight --value 0.02 \
        --key opd_consistency_lr --value 0.0001 \
        --key train --value True \
        --key val --value False \
        --key test --value True \
        --key tta_passes --value 1

export EXP_ID="$LORA_EXP_ID"
TRAIN_LLM_CONFIG="$STAGE2_CONFIG" python src/train_llm.py

LORA_DIR=$(latest_exp_dir "$LORA_EXP_ID")
echo "[llm-stage1-stage2-auto] Stage 2 output: ${LORA_DIR:-<none>}"
echo "[llm-stage1-stage2-auto] done $(date)"

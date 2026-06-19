#!/bin/bash
# Stage 2 ONLY, reusing the existing depth-24 Stage 1 final checkpoint.
# ONE-FACTOR test of the "8 tokens/video is too many" hypothesis:
#   max_frames: 8 -> 5  (=> 5 per-frame tokens/video instead of 8)
# Everything else matches the BASELINE depth-24 Stage 2 (candperm=0, no task-balance,
# 3000 steps) that scored POM acc 0.125 / slot 0.286 at 8 frames. Clean comparison.
# NOTE: encoder + Stage 1 were trained at 8 frames (frozen encoder here); this is a
# fast directional test, not a from-scratch 5-frame pipeline.

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
FRAMES="${FRAMES:-5}"
STEPS="${STEPS:-3000}"
STAGE2_CUDA="${STAGE2_CUDA:-4}"
STAGE2_GPU_CONFIG="${STAGE2_GPU_CONFIG:-configs/gpu_config_7b_gpu4.json}"
CONFIG_FILE="${CONFIG_FILE:-configs/train_llm_config.yaml}"
STAGE2_CONFIG="/tmp/octopi_stage2_depth24_frames${FRAMES}_${SEED}_${STAGE2_CUDA}.yaml"
DATA_DIR="${DATA_DIR:-/tmp/octopi_llm_frames8_encoder_full_stage12_data}"
ENCODER_PATH="${ENCODER_PATH:-exps/2026_06_07_15_08_20_train_clip_clip_seed_0_repro_sorted_valmean_ema098_frames8_depth24_gpu0/encoder.pt}"
STAGE1_DIR="${STAGE1_DIR:-exps/2026_06_07_16_25_23_train_llm_train_vicuna-7b_8000_full_pipeline_0_frames8_depth24enc_fullloss_stage1_8000_finalonly_lr2e-5_depth24enc_gpu5_now}"
LORA_EXP_ID="fp${SEED}_d24_frames${FRAMES}_${STEPS}_gpu4"

echo "[stage2-frames${FRAMES}] start $(date) | stage1=$STAGE1_DIR cuda=$STAGE2_CUDA frames=$FRAMES steps=$STEPS"
for f in "$STAGE1_DIR/final_project.pt" "$STAGE1_DIR/final_llm_weights.pt"; do
  [ -f "$f" ] || { echo "Missing Stage 1 artifact: $f"; exit 1; }
done
[ -d "$STAGE1_DIR/tokenizer" ] || { echo "Missing Stage 1 tokenizer dir"; exit 1; }
[ -f "$ENCODER_PATH" ] || { echo "Missing encoder: $ENCODER_PATH"; exit 1; }
for f in train_qa.json val_qa.json test_qa.json val_opd_qa.json test_opd_qa.json; do
  [ -f "$DATA_DIR/$f" ] || { echo "Missing QA file $DATA_DIR/$f; refusing to regenerate."; exit 1; }
done

cleanup() { echo "[stage2-frames${FRAMES}] removing private config"; rm -f "$STAGE2_CONFIG"; }
trap cleanup EXIT
set_cfg() { local cp="$1"; shift; python src/utils/update_config.py --config_path "$cp" "$@"; }
latest_exp_dir() { find exps -maxdepth 1 -type d -name "*_$1" | sort | tail -1; }

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
        --key max_frames --value "$FRAMES" \
        --key num_context_vision --value 8 \
        --key prompt_depth_vision --value 24 \
        --key use_lora --value True \
        --key lora_trained --value False \
        --key max_train_steps --value "$STEPS" \
        --key projection_path --value "$STAGE1_DIR/final_project.pt" \
        --key tokenizer_path --value "$STAGE1_DIR/tokenizer" \
        --key llm_path --value "$STAGE1_DIR/final_llm_weights.pt" \
        --key exps_path --value exps \
        --key freeze_encoder --value True \
        --key freeze_projection --value False \
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
echo "[stage2-frames${FRAMES}] Stage 2 output: ${LORA_DIR:-<none>}"
if [ -n "${LORA_DIR:-}" ] && [ -f "$LORA_DIR/test_final_results.txt" ]; then
  echo "================ RESULT (frames=$FRAMES) ================"
  grep -aiE "object_match|accuracy|slot|hardness|roughness|texture|comparison|superlative|combined" "$LORA_DIR/test_final_results.txt"
fi
echo "[stage2-frames${FRAMES}] done $(date)"

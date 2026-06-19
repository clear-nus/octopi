#!/bin/bash
# Stage 1 -> Stage 2, smart-init of ONLY the standard <tact_start>/<tact_end> tokens
# (seeded from "start"/"end" word embeddings instead of the small mean vector).
# NO distinct per-slot delimiters here -- isolates the smart-init effect first.
# Reuses the chosen f5 sinusoid encoder.
# Faithful Stage 2: q/k LoRA, llm_lr 2e-4, proj 2e-5, no aux losses, 3000 steps.
# Params: FRAMES, CUDA, ENCODER_PATH, GPU_CONFIG, LLM_DATA.

set -euo pipefail
source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi
cd /data/samson/octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

SEED="${SEED:-0}"
FRAMES="${FRAMES:-5}"
CUDA="${CUDA:-4}"
GPU_CONFIG="${GPU_CONFIG:-configs/gpu_config_7b_gpu${CUDA}.json}"
ENCODER_PATH="${ENCODER_PATH:?must set ENCODER_PATH}"
LLM_DATA="${LLM_DATA:-/tmp/octopi_llm_frames8_encoder_full_stage12_data}"
TAG="${TAG:-smartinit_se_f${FRAMES}}"

STAGE1_CFG="/tmp/octopi_s1_${TAG}_${CUDA}.yaml"
STAGE2_CFG="/tmp/octopi_s2_${TAG}_${CUDA}.yaml"
STAGE1_EXP_ID="s1_${TAG}_8000_gpu${CUDA}"
STAGE2_EXP_ID="s2_${TAG}_qk_lr2e4_3000_gpu${CUDA}"

cleanup() { echo "[${TAG}] removing private configs"; rm -f "$STAGE1_CFG" "$STAGE2_CFG"; }
trap cleanup EXIT
set_cfg() { local cp="$1"; shift; python src/utils/update_config.py --config_path "$cp" "$@"; }
latest_exp_dir() { find exps -maxdepth 1 -type d -name "*_$1" | sort | tail -1; }

[ -f "$ENCODER_PATH" ] || { echo "Missing encoder: $ENCODER_PATH"; exit 1; }
for f in train_qa.json val_qa.json test_qa.json val_opd_qa.json test_opd_qa.json; do
  [ -f "$LLM_DATA/$f" ] || { echo "Missing QA $LLM_DATA/$f"; exit 1; }
done
echo "[${TAG}] start $(date) | frames=$FRAMES cuda=$CUDA enc=$ENCODER_PATH"

# ---- Stage 1 (8000, no LoRA, train embeddings, proj/llm 2e-5) ----
echo "[${TAG}] === Stage 1 (8000) ==="
cp configs/train_llm_config.yaml "$STAGE1_CFG"
set_cfg "$STAGE1_CFG" \
        --key data_dir --value "$LLM_DATA" \
        --key train_files --value "[$LLM_DATA/train_qa.json]" \
        --key val_files --value "[$LLM_DATA/val_opd_qa.json, $LLM_DATA/val_qa.json]" \
        --key test_files --value "[$LLM_DATA/test_opd_qa.json, $LLM_DATA/test_qa.json]" \
        --key seed --value "$SEED" --key cuda --value "$CUDA" --key gpu_config --value "$GPU_CONFIG" \
        --key encoder_path --value "$ENCODER_PATH" \
        --key max_frames --value "$FRAMES" \
        --key train_all_token_embeddings --value True \
        --key smart_delimiter_init --value True \
        --key num_context_vision --value 8 --key prompt_depth_vision --value 24 \
        --key use_lora --value False --key lora_trained --value False \
        --key max_train_steps --value 8000 \
        --key val_freq --value null --key val_checkpoint_metric --value loss \
        --key projection_path --value null --key tokenizer_path --value null --key llm_path --value null \
        --key exps_path --value exps \
        --key freeze_encoder --value True --key freeze_projection --value False \
        --key modules_to_save --value "[embed_tokens]" \
        --key projection_lr --value 0.00002 --key llm_lr --value 0.00002 \
        --key warmup_steps --value 20 --key llm_gradient_accumulation_steps --value 16 \
        --key task_balanced_training --value False \
        --key conclusion_only_loss --value false --key conclusion_loss_weight --value 0.0 \
        --key conclusion_loss_task_groups --value null --key pom_conclusion_loss_weight --value null \
        --key pom_option_id_format --value false --key pom_candidate_permutation_augmentation --value 0 \
        --key multi_object_slot_permutation_augmentation --value 0 \
        --key multi_object_label_order_permutation_augmentation --value 0 \
        --key opd_consistency_loss_weight --value 0.0 --key opd_consistency_lr --value 0.0001 \
        --key train --value True --key val --value False --key test --value False --key tta_passes --value 1

EXP_ID="$STAGE1_EXP_ID" TRAIN_LLM_CONFIG="$STAGE1_CFG" python src/train_llm.py
STAGE1_DIR=$(latest_exp_dir "$STAGE1_EXP_ID")
for f in "$STAGE1_DIR/final_project.pt" "$STAGE1_DIR/final_llm_weights.pt"; do
  [ -f "$f" ] || { echo "Stage 1 artifact missing: $f"; exit 1; }
done
[ -d "$STAGE1_DIR/tokenizer" ] || { echo "Stage 1 tokenizer missing"; exit 1; }
echo "[${TAG}] Stage 1 -> $STAGE1_DIR"

# ---- Stage 2 (3000, qk LoRA, llm 2e-4, proj 2e-5, no aux) + test ----
echo "[${TAG}] === Stage 2 (3000, qk, lr2e4) ==="
cp configs/train_llm_config.yaml "$STAGE2_CFG"
set_cfg "$STAGE2_CFG" \
        --key data_dir --value "$LLM_DATA" \
        --key train_files --value "[$LLM_DATA/train_qa.json]" \
        --key val_files --value "[$LLM_DATA/val_opd_qa.json, $LLM_DATA/val_qa.json]" \
        --key test_files --value "[$LLM_DATA/test_opd_qa.json, $LLM_DATA/test_qa.json]" \
        --key seed --value "$SEED" --key cuda --value "$CUDA" --key gpu_config --value "$GPU_CONFIG" \
        --key encoder_path --value "$ENCODER_PATH" \
        --key max_frames --value "$FRAMES" \
        --key train_all_token_embeddings --value True \
        --key smart_delimiter_init --value True \
        --key num_context_vision --value 8 --key prompt_depth_vision --value 24 \
        --key use_lora --value True --key lora_trained --value False \
        --key max_train_steps --value 3000 \
        --key projection_path --value "$STAGE1_DIR/final_project.pt" \
        --key tokenizer_path --value "$STAGE1_DIR/tokenizer" \
        --key llm_path --value "$STAGE1_DIR/final_llm_weights.pt" \
        --key exps_path --value exps \
        --key freeze_encoder --value True --key freeze_projection --value False \
        --key modules_to_save --value "[embed_tokens]" \
        --key projection_lr --value 0.00002 --key llm_lr --value 0.0002 \
        --key warmup_steps --value 50 \
        --key val_freq --value null --key val_checkpoint_metric --value loss \
        --key r --value 128 --key lora_alpha --value 256 --key lora_dropout --value 0.05 \
        --key target_modules --value "[q_proj, k_proj]" \
        --key llm_gradient_accumulation_steps --value 16 \
        --key task_balanced_training --value False \
        --key conclusion_only_loss --value false --key conclusion_loss_weight --value 0.0 \
        --key conclusion_loss_task_groups --value null --key pom_conclusion_loss_weight --value null \
        --key pom_option_id_format --value false --key pom_candidate_permutation_augmentation --value 0 \
        --key multi_object_slot_permutation_augmentation --value 0 \
        --key multi_object_label_order_permutation_augmentation --value 0 \
        --key opd_consistency_loss_weight --value 0.0 --key opd_consistency_lr --value 0.0001 \
        --key train --value True --key val --value False --key test --value True --key tta_passes --value 1

EXP_ID="$STAGE2_EXP_ID" TRAIN_LLM_CONFIG="$STAGE2_CFG" python src/train_llm.py
STAGE2_DIR=$(latest_exp_dir "$STAGE2_EXP_ID")
echo "[${TAG}] Stage 2 -> ${STAGE2_DIR:-<none>}"
RES="$STAGE2_DIR/test_final_partial_results.txt"; [ -f "$RES" ] || RES="$STAGE2_DIR/test_final_results.txt"
if [ -f "$RES" ]; then
  echo "================ RESULT (${TAG}) ================"
  grep -aiE "object_match|slot|hardness|roughness|texture|comparison|superlative|combined|accuracy" "$RES"
fi
echo "[${TAG}] done $(date)"

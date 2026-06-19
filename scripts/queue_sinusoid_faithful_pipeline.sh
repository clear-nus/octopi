#!/bin/bash
# Faithful test of the missing per-frame sinusoidal positional embedding.
# Re-adds the original Octopi per-frame sinusoid (now live in src/utils/model.py:
#   ViFiCLIP.forward and MultimodalLLMForCausalLM.forward) and retrains the FULL
#   pipeline: CLIP encoder -> Stage 1 -> Stage 2.
#
# Encoder keeps OUR improvements (lr 3e-4, EMA 0.98, scaled CBS, decoupled heads,
#   depth_vision=24, val_mean selection) + the sinusoid. Stage 2 uses the ORIGINAL
#   LoRA LR (2e-4), q/k targets, no aux losses (paper-faithful Stage 2).
#
# Parametrized by FRAMES so we can run 5-frame and 8-frame encoders in parallel.
# Reuses already-processed frames (loader sub-samples max_frames at load time);
# does NOT reprocess or regenerate QA.

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
FRAMES="${FRAMES:-8}"
CUDA="${CUDA:-3}"
GPU_CONFIG="${GPU_CONFIG:-configs/gpu_config_7b_gpu${CUDA}.json}"
CLIP_DATA="${CLIP_DATA:-/tmp/octopi_clip_frames8_depth24_gpu0/seed_0}"
LLM_DATA="${LLM_DATA:-/tmp/octopi_llm_frames8_encoder_full_stage12_data}"

CLIP_CFG="/tmp/octopi_clip_sinusoid_f${FRAMES}_${SEED}_${CUDA}.yaml"
STAGE1_CFG="/tmp/octopi_s1_sinusoid_f${FRAMES}_${SEED}_${CUDA}.yaml"
STAGE2_CFG="/tmp/octopi_s2_sinusoid_f${FRAMES}_${SEED}_${CUDA}.yaml"

CLIP_EXP_ID="clip_s0_sinusoid_f${FRAMES}_d24_gpu${CUDA}"
STAGE1_EXP_ID="s1_sinusoid_f${FRAMES}_8000_gpu${CUDA}"
STAGE2_EXP_ID="s2_sinusoid_f${FRAMES}_qk_lr2e4_3000_gpu${CUDA}"

cleanup() { echo "[sinusoid-f${FRAMES}] removing private configs"; rm -f "$CLIP_CFG" "$STAGE1_CFG" "$STAGE2_CFG"; }
trap cleanup EXIT
set_cfg() { local cp="$1"; shift; python src/utils/update_config.py --config_path "$cp" "$@"; }
latest_exp_dir() { find exps -maxdepth 1 -type d -name "*_$1" | sort | tail -1; }

echo "[sinusoid-f${FRAMES}] start $(date) | frames=$FRAMES cuda=$CUDA"
# sanity: data present
[ -f "$CLIP_DATA/train_samples.json" ] || { echo "Missing CLIP samples in $CLIP_DATA"; exit 1; }
for f in train_qa.json val_qa.json test_qa.json val_opd_qa.json test_opd_qa.json; do
  [ -f "$LLM_DATA/$f" ] || { echo "Missing QA file $LLM_DATA/$f; refusing to regenerate."; exit 1; }
done

# ----------------------------------------------------------------------------
# 1) CLIP encoder retrain (our improved recipe + per-frame sinusoid)
# ----------------------------------------------------------------------------
echo "[sinusoid-f${FRAMES}] === CLIP encoder retrain ==="
cp configs/train_clip_config.yaml "$CLIP_CFG"
set_cfg "$CLIP_CFG" \
        --key data_dir --value "$CLIP_DATA" \
        --key cuda --value "$CUDA" \
        --key seed --value "$SEED" \
        --key num_epochs --value 30 \
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
        --key swa --value false \
        --key flip_p --value 0.5 \
        --key rotation_degrees --value 0 \
        --key color_jitter --value 0.0 \
        --key gaussian_blur --value false

EXP_ID="$CLIP_EXP_ID" TRAIN_CLIP_CONFIG="$CLIP_CFG" python src/train_clip.py

CLIP_DIR=$(latest_exp_dir "$CLIP_EXP_ID")
ENCODER_PATH="$CLIP_DIR/encoder.pt"
[ -f "$ENCODER_PATH" ] || { echo "CLIP encoder missing: $ENCODER_PATH"; exit 1; }
echo "[sinusoid-f${FRAMES}] encoder -> $ENCODER_PATH"

# ----------------------------------------------------------------------------
# 2) Stage 1 (alignment, no LoRA, full answer loss, train embeddings, 8000)
# ----------------------------------------------------------------------------
echo "[sinusoid-f${FRAMES}] === Stage 1 (8000) ==="
cp configs/train_llm_config.yaml "$STAGE1_CFG"
set_cfg "$STAGE1_CFG" \
        --key data_dir --value "$LLM_DATA" \
        --key train_files --value "[$LLM_DATA/train_qa.json]" \
        --key val_files --value "[$LLM_DATA/val_opd_qa.json, $LLM_DATA/val_qa.json]" \
        --key test_files --value "[$LLM_DATA/test_opd_qa.json, $LLM_DATA/test_qa.json]" \
        --key seed --value "$SEED" \
        --key cuda --value "$CUDA" \
        --key gpu_config --value "$GPU_CONFIG" \
        --key encoder_path --value "$ENCODER_PATH" \
        --key max_frames --value "$FRAMES" \
        --key train_all_token_embeddings --value True \
        --key num_context_vision --value 8 \
        --key prompt_depth_vision --value 24 \
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

EXP_ID="$STAGE1_EXP_ID" TRAIN_LLM_CONFIG="$STAGE1_CFG" python src/train_llm.py
STAGE1_DIR=$(latest_exp_dir "$STAGE1_EXP_ID")
for f in "$STAGE1_DIR/final_project.pt" "$STAGE1_DIR/final_llm_weights.pt"; do
  [ -f "$f" ] || { echo "Stage 1 artifact missing: $f"; exit 1; }
done
[ -d "$STAGE1_DIR/tokenizer" ] || { echo "Stage 1 tokenizer missing"; exit 1; }
echo "[sinusoid-f${FRAMES}] Stage 1 -> $STAGE1_DIR"

# ----------------------------------------------------------------------------
# 3) Stage 2 (LoRA q/k, original LLM LR 2e-4, proj 2e-5, no aux, 3000) + test
# ----------------------------------------------------------------------------
echo "[sinusoid-f${FRAMES}] === Stage 2 (3000, qk, lr2e4) ==="
cp configs/train_llm_config.yaml "$STAGE2_CFG"
set_cfg "$STAGE2_CFG" \
        --key data_dir --value "$LLM_DATA" \
        --key train_files --value "[$LLM_DATA/train_qa.json]" \
        --key val_files --value "[$LLM_DATA/val_opd_qa.json, $LLM_DATA/val_qa.json]" \
        --key test_files --value "[$LLM_DATA/test_opd_qa.json, $LLM_DATA/test_qa.json]" \
        --key seed --value "$SEED" \
        --key cuda --value "$CUDA" \
        --key gpu_config --value "$GPU_CONFIG" \
        --key encoder_path --value "$ENCODER_PATH" \
        --key max_frames --value "$FRAMES" \
        --key train_all_token_embeddings --value True \
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
        --key llm_lr --value 0.0002 \
        --key warmup_steps --value 50 \
        --key val_freq --value null \
        --key val_checkpoint_metric --value loss \
        --key r --value 128 \
        --key lora_alpha --value 256 \
        --key lora_dropout --value 0.05 \
        --key target_modules --value "[q_proj, k_proj]" \
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

EXP_ID="$STAGE2_EXP_ID" TRAIN_LLM_CONFIG="$STAGE2_CFG" python src/train_llm.py
STAGE2_DIR=$(latest_exp_dir "$STAGE2_EXP_ID")
echo "[sinusoid-f${FRAMES}] Stage 2 -> ${STAGE2_DIR:-<none>}"
RES="$STAGE2_DIR/test_final_partial_results.txt"
[ -f "$RES" ] || RES="$STAGE2_DIR/test_final_results.txt"
if [ -f "$RES" ]; then
  echo "================ RESULT (frames=$FRAMES, sinusoid) ================"
  grep -aiE "object_match|slot|hardness|roughness|texture|comparison|superlative|combined|accuracy" "$RES"
fi
echo "[sinusoid-f${FRAMES}] done $(date)"

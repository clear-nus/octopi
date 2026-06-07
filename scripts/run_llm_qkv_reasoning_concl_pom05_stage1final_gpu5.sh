#!/bin/bash
# q/k/v LoRA using the Stage 1 final 8000-step checkpoint.
# Reuses prepared QA files; does not regenerate data.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

cd /data/samson/octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=0
CUDA=5
CONFIG_FILE="configs/train_llm_config.yaml"
RUN_CONFIG="/tmp/octopi_train_llm_qkv_reasonconcl_pom05_stage1final_gpu5_config.yaml"
DATA_DIR="/tmp/octopi_llm_frames8_encoder_full_stage12_data"
ENCODER_PATH="exps/2026_06_01_15_28_38_train_clip_clip_seed_0_repro_sorted_valmean_ema098_frames8_notopk/encoder.pt"
STAGE1_DIR="exps/2026_06_07_07_25_52_train_llm_train_vicuna-7b_8000_full_pipeline_0_frames8enc_fullloss_stage1_8000_finalonly_lr2e-5_gpu4"
LORA_EXP_ID="full_pipeline_${SEED}_frames8enc_fullloss_stage1_8000final_lora_qkv_r128_a256_lr0.0001_freezeproj_finalonly_originalpom_reasonconcl0.25_pom0.5_cons0.02_3000_gpu5"

echo "[llm-qkv-stage1final-gpu5] start $(date)"

cp "$CONFIG_FILE" "$RUN_CONFIG"
trap 'echo "[llm-qkv-stage1final-gpu5] removing private config"; rm -f "$RUN_CONFIG"' EXIT

set_cfg() {
  python src/utils/update_config.py --config_path "$RUN_CONFIG" "$@"
}

latest_exp_dir() {
  local exp_id="$1"
  find exps -maxdepth 1 -type d -name "*_${exp_id}" | sort | tail -1
}

if [ ! -f "$ENCODER_PATH" ]; then
  echo "Missing encoder checkpoint: $ENCODER_PATH"
  exit 1
fi

if [ ! -f "$STAGE1_DIR/final_project.pt" ] || [ ! -f "$STAGE1_DIR/final_llm_weights.pt" ] || [ ! -d "$STAGE1_DIR/tokenizer" ]; then
  echo "Stage 1 final output missing required artifacts: $STAGE1_DIR"
  exit 1
fi

if [ ! -f "$DATA_DIR/train_qa.json" ] || [ ! -f "$DATA_DIR/val_qa.json" ] || [ ! -f "$DATA_DIR/test_qa.json" ] || [ ! -f "$DATA_DIR/val_opd_qa.json" ] || [ ! -f "$DATA_DIR/test_opd_qa.json" ]; then
  echo "Missing existing QA files in $DATA_DIR; refusing to process/regenerate data in this script."
  exit 1
fi

echo "[llm-qkv-stage1final-gpu5] Stage 2: q/k/v LoRA from Stage 1 final, original POM, reasoning conclusion=0.25, POM conclusion=0.5, OPD consistency=0.02, final-only eval, steps=3000"
set_cfg --key data_dir --value "$DATA_DIR" \
        --key train_files --value "[$DATA_DIR/train_qa.json]" \
        --key val_files --value "[$DATA_DIR/val_opd_qa.json, $DATA_DIR/val_qa.json]" \
        --key test_files --value "[$DATA_DIR/test_opd_qa.json, $DATA_DIR/test_qa.json]" \
        --key seed --value "$SEED" \
        --key cuda --value "$CUDA" \
        --key gpu_config --value configs/gpu_config_7b_gpu5.json \
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
TRAIN_LLM_CONFIG="$RUN_CONFIG" python src/train_llm.py

echo "[llm-qkv-stage1final-gpu5] completed LoRA: $(latest_exp_dir "$LORA_EXP_ID")"
echo "[llm-qkv-stage1final-gpu5] done $(date)"

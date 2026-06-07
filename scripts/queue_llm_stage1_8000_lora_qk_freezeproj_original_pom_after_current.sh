#!/bin/bash
# Original POM answer-format control after the current q/k/v/o lower-LR run.
# Keeps the strong q/k frozen-projector setup, but disables option-ID POM training.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

cd /data/samson/octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WAIT_SESSION="octopi_llm_pomopt_qkvo_lr1e4"
SEED=0
CUDA=6
CONFIG_FILE="configs/train_llm_config.yaml"
CONFIG_BAK="${CONFIG_FILE}.stage1_8000_qk_freezeproj_original_pom_bak"
DATA_DIR="/tmp/octopi_llm_frames8_encoder_full_stage12_data"
ENCODER_PATH="exps/2026_06_01_15_28_38_train_clip_clip_seed_0_repro_sorted_valmean_ema098_frames8_notopk/encoder.pt"
STAGE1_DIR="exps/2026_06_02_11_37_05_train_llm_train_val_test_vicuna-7b_8000_full_pipeline_0_frames8enc_fullloss_stage1_8000_paperaligned"
LORA_EXP_ID="full_pipeline_${SEED}_frames8enc_fullloss_stage1_8000_lora_qk_r128_a256_lr0.0002_freezeproj_lossval_originalpom"

echo "[llm-qk-original-pom] queued $(date)"
echo "[llm-qk-original-pom] waiting for tmux session ${WAIT_SESSION} to finish"
while tmux has-session -t "=${WAIT_SESSION}" 2>/dev/null; do
  sleep 120
done

echo "[llm-qk-original-pom] start $(date)"

if pgrep -f "python.*src/train_llm.py" >/dev/null; then
  echo "Another train_llm.py process is active; refusing to start a duplicate run."
  exit 1
fi

cp "$CONFIG_FILE" "$CONFIG_BAK"
trap 'echo "[llm-qk-original-pom] restoring config"; cp "$CONFIG_BAK" "$CONFIG_FILE"; rm -f "$CONFIG_BAK"' EXIT

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

if [ ! -f "$STAGE1_DIR/best_project.pt" ] || [ ! -f "$STAGE1_DIR/best_llm_weights.pt" ] || [ ! -d "$STAGE1_DIR/tokenizer" ]; then
  echo "Stage 1 output missing required artifacts: $STAGE1_DIR"
  exit 1
fi

if [ ! -f "$DATA_DIR/train_qa.json" ] || [ ! -f "$DATA_DIR/val_qa.json" ] || [ ! -f "$DATA_DIR/test_qa.json" ] || [ ! -f "$DATA_DIR/val_opd_qa.json" ] || [ ! -f "$DATA_DIR/test_opd_qa.json" ]; then
  echo "Missing existing QA files in $DATA_DIR; refusing to process/regenerate data in this script."
  exit 1
fi

echo "[llm-qk-original-pom] Stage 2: q/k LoRA, frozen projector, loss checkpoint, original POM object-name format"
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
        --key freeze_projection --value True \
        --key modules_to_save --value "[embed_tokens]" \
        --key projection_lr --value 0.00002 \
        --key llm_lr --value 0.0002 \
        --key warmup_steps --value 20 \
        --key val_freq --value 150 \
        --key val_checkpoint_metric --value loss \
        --key r --value 128 \
        --key lora_alpha --value 256 \
        --key lora_dropout --value 0.05 \
        --key target_modules --value "[q_proj, k_proj]" \
        --key llm_gradient_accumulation_steps --value 16 \
        --key conclusion_only_loss --value false \
        --key pom_option_id_format --value false \
        --key pom_candidate_permutation_augmentation --value 0 \
        --key opd_consistency_loss_weight --value 0.0 \
        --key train --value True \
        --key val --value True \
        --key test --value True \
        --key tta_passes --value 1

export EXP_ID="$LORA_EXP_ID"
python src/train_llm.py

echo "[llm-qk-original-pom] completed LoRA: $(latest_exp_dir "$LORA_EXP_ID")"
echo "[llm-qk-original-pom] done $(date)"

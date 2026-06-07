#!/bin/bash
# Test-only evaluation for the stopped OPD-consistency ablation's saved best checkpoint.

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
CUDA=6
CONFIG_FILE="configs/train_llm_config.yaml"
CONFIG_BAK="${CONFIG_FILE}.opdcons005_test_only_bak"
DATA_DIR="/tmp/octopi_llm_frames8_encoder_full_stage12_data"
ENCODER_PATH="exps/2026_06_01_15_28_38_train_clip_clip_seed_0_repro_sorted_valmean_ema098_frames8_notopk/encoder.pt"
SOURCE_EXP="exps/2026_06_03_05_54_49_train_llm_train_val_test_lora_256_128_vicuna-7b_3000_full_pipeline_0_frames8enc_fullloss_stage1_8000_lora_qk_r128_a256_lr0.0001_freezeproj_opdcons005"
LORA_PATH="${SOURCE_EXP}/best_llm_weights"
PROJECT_PATH="${SOURCE_EXP}/best_project.pt"
EXP_SUFFIX="opdcons005_best_test_only"

echo "[llm-opdcons005-test] start $(date)"

if pgrep -f "python.*src/train_llm.py" >/dev/null; then
  echo "Another train_llm.py process is active; refusing to start a duplicate run."
  exit 1
fi

cp "$CONFIG_FILE" "$CONFIG_BAK"
trap 'echo "[llm-opdcons005-test] restoring config"; cp "$CONFIG_BAK" "$CONFIG_FILE"; rm -f "$CONFIG_BAK"' EXIT

set_cfg() {
  python src/utils/update_config.py --config_path "$CONFIG_FILE" "$@"
}

if [ ! -f "$ENCODER_PATH" ]; then
  echo "Missing encoder checkpoint: $ENCODER_PATH"
  exit 1
fi

if [ ! -f "$PROJECT_PATH" ] || [ ! -f "$LORA_PATH/adapter_model.bin" ] || [ ! -f "$LORA_PATH/adapter_config.json" ]; then
  echo "Missing saved best checkpoint artifacts under $SOURCE_EXP"
  exit 1
fi

if [ ! -f "$DATA_DIR/test_qa.json" ] || [ ! -f "$DATA_DIR/test_opd_qa.json" ]; then
  echo "Missing existing test QA files in $DATA_DIR; refusing to process/regenerate data in this script."
  exit 1
fi

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
        --key projection_path --value "$PROJECT_PATH" \
        --key tokenizer_path --value "$SOURCE_EXP/tokenizer" \
        --key llm_path --value "$LORA_PATH" \
        --key exps_path --value exps \
        --key freeze_encoder --value True \
        --key freeze_projection --value True \
        --key modules_to_save --value "[embed_tokens]" \
        --key projection_lr --value 0.00002 \
        --key llm_lr --value 0.0001 \
        --key warmup_steps --value 20 \
        --key val_freq --value 150 \
        --key r --value 128 \
        --key lora_alpha --value 256 \
        --key lora_dropout --value 0.05 \
        --key target_modules --value "[q_proj, k_proj]" \
        --key llm_gradient_accumulation_steps --value 16 \
        --key conclusion_only_loss --value false \
        --key opd_consistency_loss_weight --value 0.0 \
        --key train --value False \
        --key val --value False \
        --key test --value True \
        --key tta_passes --value 1

export EXP_ID="$EXP_SUFFIX"
python src/train_llm.py

echo "[llm-opdcons005-test] done $(date)"

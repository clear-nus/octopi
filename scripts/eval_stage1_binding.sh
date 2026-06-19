#!/bin/bash
# Test-only eval of a Stage-1 (alignment-only, NO LoRA) checkpoint, to measure whether
# binding (POM) is already present after alignment, before the Stage-2 q/k LoRA.
# Single eval pass (train=False -> only test_best, no redundant test_final).
# Params: STAGE1_DIR, ENCODER_PATH, FRAMES, CUDA, DISTINCT, LLM_DATA.

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

STAGE1_DIR="${STAGE1_DIR:?must set STAGE1_DIR}"
ENCODER_PATH="${ENCODER_PATH:?must set ENCODER_PATH}"
FRAMES="${FRAMES:-5}"
CUDA="${CUDA:-6}"
DISTINCT="${DISTINCT:-True}"
GPU_CONFIG="${GPU_CONFIG:-configs/gpu_config_7b_gpu${CUDA}.json}"
LLM_DATA="${LLM_DATA:-/tmp/octopi_llm_frames8_encoder_full_stage12_data}"
TAG="${TAG:-s1eval_distinctdelim_f${FRAMES}}"

CFG="/tmp/octopi_${TAG}_${CUDA}.yaml"
EXP_ID="${TAG}_gpu${CUDA}"
cleanup() { rm -f "$CFG"; }
trap cleanup EXIT
set_cfg() { local cp="$1"; shift; python src/utils/update_config.py --config_path "$cp" "$@"; }
latest_exp_dir() { find exps -maxdepth 1 -type d -name "*_$1" | sort | tail -1; }

for f in "$STAGE1_DIR/final_project.pt" "$STAGE1_DIR/final_llm_weights.pt"; do
  [ -f "$f" ] || { echo "Missing Stage 1 artifact: $f"; exit 1; }
done
[ -d "$STAGE1_DIR/tokenizer" ] || { echo "Missing Stage 1 tokenizer"; exit 1; }
[ -f "$ENCODER_PATH" ] || { echo "Missing encoder: $ENCODER_PATH"; exit 1; }
echo "[${TAG}] start $(date) | stage1=$STAGE1_DIR distinct=$DISTINCT"

cp configs/train_llm_config.yaml "$CFG"
set_cfg "$CFG" \
        --key data_dir --value "$LLM_DATA" \
        --key train_files --value "[$LLM_DATA/train_qa.json]" \
        --key val_files --value "[$LLM_DATA/val_opd_qa.json, $LLM_DATA/val_qa.json]" \
        --key test_files --value "[$LLM_DATA/test_opd_qa.json, $LLM_DATA/test_qa.json]" \
        --key seed --value 0 --key cuda --value "$CUDA" --key gpu_config --value "$GPU_CONFIG" \
        --key encoder_path --value "$ENCODER_PATH" \
        --key max_frames --value "$FRAMES" \
        --key train_all_token_embeddings --value True \
        --key distinct_delimiters --value "$DISTINCT" \
        --key num_context_vision --value 8 --key prompt_depth_vision --value 24 \
        --key use_lora --value False --key lora_trained --value False \
        --key projection_path --value "$STAGE1_DIR/final_project.pt" \
        --key tokenizer_path --value "$STAGE1_DIR/tokenizer" \
        --key llm_path --value "$STAGE1_DIR/final_llm_weights.pt" \
        --key exps_path --value exps \
        --key freeze_encoder --value True --key freeze_projection --value True \
        --key val_freq --value null --key val_checkpoint_metric --value loss \
        --key train --value False --key val --value False --key test --value True --key tta_passes --value 1

EXP_ID="$EXP_ID" TRAIN_LLM_CONFIG="$CFG" python src/train_llm.py
DIR=$(latest_exp_dir "$EXP_ID")
RES="$DIR/test_best_results.txt"; [ -f "$RES" ] || RES="$DIR/test_best_partial_results.txt"
echo "[${TAG}] dir -> ${DIR:-<none>}"
if [ -f "$RES" ]; then
  echo "================ STAGE 1 BINDING RESULT (${TAG}) ================"
  grep -aiE "object_match|slot|hardness|roughness|texture|comparison|superlative|combined|accuracy" "$RES"
fi
echo "[${TAG}] done $(date)"

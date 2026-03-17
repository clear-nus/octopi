#!/bin/bash

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Configuration
CONFIG_FILE="configs/train_llm_config.yaml"
ENCODER_PATH="exps/2026_02_02_10_44_30_train_clip_clip_seed_0/encoder.pt"
STAGE1_EXP_DIR="exps/2026_02_02_11_18_05_train_llm_train_val_test_vicuna-7b_8000_full_pipeline_0"

# Grid Search Parameters

LLM_GRADIENT_ACCUM_STEPS=(16)
LORA_DROPOUT=(0.05)

echo "----------------------------------------------------------------"
echo "Starting LoRA Grid Search"
echo "Stage 1 Weights: $STAGE1_EXP_DIR"
echo "----------------------------------------------------------------"

# Generate Data
DATA_DIR="data"
DATASET_PATH="dataset"
SEED=0
echo "Cleaning up data directory..."
rm -rf "$DATA_DIR"
echo "Generating data for seed $SEED in $DATA_DIR..."
python src/utils/process_dataset.py --dataset_path "$DATASET_PATH" --output_path "$DATA_DIR" --seed "$SEED"
python src/utils/generate_qa.py --data_path "$DATA_DIR" --seed "$SEED"

for LLM_GRADIENT_ACCUM_STEPS_VAL in "${LLM_GRADIENT_ACCUM_STEPS[@]}"; do
    for LORA_DROPOUT_VAL in "${LORA_DROPOUT[@]}"; do
        
        echo "----------------------------------------------------------------"
        echo "Running LoRA Config: llm_gradient_accumulation_steps=$LLM_GRADIENT_ACCUM_STEPS_VAL, lora_dropout=$LORA_DROPOUT_VAL"
        echo "----------------------------------------------------------------"

        # Update Config for this run
        # We ensure we point to the Stage 1 weights and set LoRA params
        python src/utils/update_config.py --config_path "$CONFIG_FILE" \
            --key use_lora --value True \
            --key encoder_path --value "$ENCODER_PATH" \
            --key lora_trained --value False \
            --key max_train_steps --value 3000 \
            --key val_freq --value 300 \
            --key projection_path --value "$STAGE1_EXP_DIR/best_project.pt" \
            --key tokenizer_path --value "$STAGE1_EXP_DIR/tokenizer" \
            --key llm_path --value "$STAGE1_EXP_DIR/best_llm_weights.pt" \
            --key exps_path --value exps \
            --key freeze_projection --value False \
            --key modules_to_save --value "[embed_tokens]" \
            --key projection_lr --value 0.0002 \
            --key llm_lr --value 0.0002 \
            --key warmup_steps --value 0.03 \
            --key r --value 128 \
            --key lora_alpha --value 256 \
            --key lora_dropout --value "$LORA_DROPOUT_VAL" \
            --key llm_gradient_accumulation_steps --value "$LLM_GRADIENT_ACCUM_STEPS_VAL"

        # Set Experiment ID
        export EXP_ID="lora_grid"
        
        echo "Running train_llm.py..."
        python src/train_llm.py

        # Find output dir
        LATEST_EXP_DIR=$(ls -td exps/*_$EXP_ID | head -1)
        echo "Experiment completed. Output: $LATEST_EXP_DIR"
        echo ""
    done
done

echo "LoRA Grid Search Completed."
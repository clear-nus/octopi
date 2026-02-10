#!/bin/bash

# Array of seeds to test
SEEDS=(0)

# Path to config file
CONFIG_FILE="configs/train_llm_config.yaml"
DATASET_PATH="dataset"
DATA_DIR="data"

for SEED in "${SEEDS[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Running LLM training experiment with seed: $SEED"
    echo "----------------------------------------------------------------"

    # 1. Generate Data
    echo "Cleaning up data directory..."
    rm -rf "$DATA_DIR"
    echo "Generating data for seed $SEED in $DATA_DIR..."
    python utils/process_dataset.py --dataset_path "$DATASET_PATH" --output_path "$DATA_DIR" --seed "$SEED"
    python utils/generate_qa.py --data_path "$DATA_DIR" --seed "$SEED"

    # # 1.5 Train Encoder
    # CLIP_CONFIG_FILE="configs/train_clip_config.yaml"
    # echo "Training Encoder..."
    # python utils/update_config.py --config_path "$CLIP_CONFIG_FILE" \
    #     --key data_dir --value "$DATA_DIR" \
    #     --key seed --value "$SEED"
    
    export EXP_ID="clip_seed_${SEED}"
    # python train_clip.py
    
    # Find encoder output
    LATEST_CLIP_DIR=$(ls -td exps/*_$EXP_ID | head -1)
    ENCODER_PATH="$LATEST_CLIP_DIR/encoder.pt"
    echo "Encoder training completed. Encoder at: $ENCODER_PATH"

    # 2. Run Training (Stage 1)
    # Reset Stage 1 config
    python utils/update_config.py --config_path "$CONFIG_FILE" \
        --key seed --value "$SEED" \
        --key encoder_path --value "$ENCODER_PATH" \
        --key use_lora --value False \
        --key max_train_steps --value 8000 \
        --key val_freq --value 2000 \
        --key projection_path --value null \
        --key tokenizer_path --value null \
        --key llm_path --value null \
        --key exps_path --value exps \
        --key freeze_projection --value False \
        --key modules_to_save --value "[embed_tokens]" \
        --key projection_lr --value 0.00002 \
        --key llm_lr --value 0.0002 \
        --key warmup_steps --value 0.03 \
        --key llm_gradient_accumulation_steps --value 16

    # Pass seed as experiment identifier
    export EXP_ID="full_pipeline_${SEED}"
    
    # echo "Running train_llm.py (Stage 1)..."
    # python train_llm.py
    
    # Find output dir of Stage 1
    # We look for the most recent directory ending in _full_pipeline_$SEED
    LATEST_EXP_DIR=$(ls -td exps/*_full_pipeline_$SEED | head -1)
    echo "Stage 1 completed. Output: $LATEST_EXP_DIR"

    # 4. Run LoRA Finetuning (Stage 2)
    echo "Setting up LoRA finetuning..."
    
    # Update config for LoRA
    python utils/update_config.py --config_path "$CONFIG_FILE" \
        --key seed --value "$SEED" \
        --key encoder_path --value "$ENCODER_PATH" \
        --key use_lora --value True \
        --key lora_trained --value False \
        --key max_train_steps --value 3000 \
        --key val_freq --value 300 \
        --key projection_path --value "$LATEST_EXP_DIR/best_project.pt" \
        --key tokenizer_path --value "$LATEST_EXP_DIR/tokenizer" \
        --key llm_path --value "$LATEST_EXP_DIR/best_llm_weights.pt" \
        --key exps_path --value exps \
        --key freeze_projection --value False \
        --key modules_to_save --value "[embed_tokens]" \
        --key projection_lr --value 0.0002 \
        --key llm_lr --value 0.0002 \
        --key warmup_steps --value 0.05 \
        --key r --value 128 \
        --key lora_alpha --value 256 \
        --key lora_dropout --value 0.05 \
        --key llm_gradient_accumulation_steps --value 16
    
    export EXP_ID="full_pipeline_${SEED}_lora"
    echo "Running train_llm.py (Stage 2 - LoRA)..."
    python train_llm.py

    # Find output dir of Stage 2
    LATEST_LORA_EXP_DIR=$(ls -td exps/*_${SEED}_lora | head -1)
    echo "Stage 2 completed. Output: $LATEST_LORA_EXP_DIR"

    echo "Finished runs for seed $SEED"
    echo ""
done

echo "All LLM training reproducibility runs completed."

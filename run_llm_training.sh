#!/bin/bash

# Array of seeds to test
SEEDS=(0 1 2 3 4)

# Path to config file
CONFIG_FILE="configs/train_llm_config.yaml"
DATASET_PATH="dataset"

for SEED in "${SEEDS[@]}"; do
    DATA_DIR="data"
    
    echo "----------------------------------------------------------------"
    echo "Running LLM training experiment with seed: $SEED"
    echo "----------------------------------------------------------------"

    # 1. Generate Data
    echo "Cleaning up data directory..."
    rm -rf "$DATA_DIR"
    echo "Generating data for seed $SEED in $DATA_DIR..."
    python utils/process_dataset.py --dataset_path "$DATASET_PATH" --output_path "$DATA_DIR" --seed "$SEED"
    python utils/generate_qa.py --data_path "$DATA_DIR" --seed "$SEED"

    # 2. Update Config
    # Update seed and encoder path
    python utils/update_config.py --config_path "$CONFIG_FILE" \
        --key seed --value "$SEED" \
        --key encoder_path --value "/data/samson/octopi/exps/train_clip_seed_$SEED/encoder.pt"
    
    echo "Updated $CONFIG_FILE with seed $SEED, data path $DATA_DIR, and encoder path"

    # 3. Run Training (Stage 1)
    # Reset Stage 1 config
    python utils/update_config.py --config_path "$CONFIG_FILE" \
        --key use_lora --value False \
        --key max_train_steps --value 8000 \
        --key val_freq --value 800 \
        --key projection_path --value null \
        --key tokenizer_path --value null \
        --key llm_path --value null \
        --key exps_path --value exps \
        --key freeze_projection --value False \
        --key modules_to_save --value "[embed_tokens]" \
        --key projection_lr --value 0.001 \
        --key llm_lr --value 0.001 \
        --key warmup_steps --value 0.03

    # Pass seed as experiment identifier
    export EXP_ID="$SEED"
    
    echo "Running train_llm.py (Stage 1)..."
    python train_llm.py
    
    # Find output dir of Stage 1
    # We look for the most recent directory ending in _$SEED
    LATEST_EXP_DIR=$(ls -td exps/*_$SEED | head -1)
    echo "Stage 1 completed. Output: $LATEST_EXP_DIR"

    # Evaluate Stage 1
    if [ -f "$LATEST_EXP_DIR/test_preds.json" ]; then
        echo "Evaluating Stage 1 results..."
        python evaluate_llm.py --test_preds_path "$LATEST_EXP_DIR/test_preds.json" > "$LATEST_EXP_DIR/evaluation_results.txt"
        cat "$LATEST_EXP_DIR/evaluation_results.txt"
    else
        echo "Warning: Stage 1 test_preds.json not found."
    fi

    # 4. Run LoRA Finetuning (Stage 2)
    echo "Setting up LoRA finetuning..."
    
    # Update config for LoRA
    python utils/update_config.py --config_path "$CONFIG_FILE" \
        --key use_lora --value True \
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
        --key warmup_steps --value 0.1 \
        --key r --value 128 \
        --key lora_alpha --value 256 \
        --key lora_dropout --value 0.1
    
    export EXP_ID="${SEED}_lora"
    echo "Running train_llm.py (Stage 2 - LoRA)..."
    python train_llm.py

    # Find output dir of Stage 2
    LATEST_LORA_EXP_DIR=$(ls -td exps/*_${SEED}_lora | head -1)
    echo "Stage 2 completed. Output: $LATEST_LORA_EXP_DIR"

    # Evaluate Stage 2 
    if [ -f "$LATEST_LORA_EXP_DIR/test_preds.json" ]; then
        echo "Evaluating Stage 2 results..."
        python evaluate_llm.py --test_preds_path "$LATEST_LORA_EXP_DIR/test_preds.json" > "$LATEST_LORA_EXP_DIR/evaluation_results.txt"
        cat "$LATEST_LORA_EXP_DIR/evaluation_results.txt"
    else
        echo "Warning: Stage 2 test_preds.json not found."
    fi

    echo "Finished runs for seed $SEED"
    echo ""
done

echo "All LLM training reproducibility runs completed."

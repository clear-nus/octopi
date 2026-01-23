#!/bin/bash

# Configuration
CONFIG_FILE="configs/train_llm_config.yaml"
STAGE1_EXP_DIR="exps/2026_01_05_23_39_30_train_llm_train_val_test_vicuna-7b_6000_0"

# Grid Search Parameters

LLM_GRADIENT_ACCUM_STEPS=(64)
LORA_DROPOUT=(0.1 0.2 0.3)

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
python utils/process_dataset.py --dataset_path "$DATASET_PATH" --output_path "$DATA_DIR" --seed "$SEED"
python utils/generate_qa.py --data_path "$DATA_DIR" --seed "$SEED"

for LLM_GRADIENT_ACCUM_STEPS_VAL in "${LLM_GRADIENT_ACCUM_STEPS[@]}"; do
    for LORA_DROPOUT_VAL in "${LORA_DROPOUT[@]}"; do
        
        echo "----------------------------------------------------------------"
        echo "Running LoRA Config: llm_gradient_accumulation_steps=$LLM_GRADIENT_ACCUM_STEPS_VAL, lora_dropout=$LORA_DROPOUT_VAL"
        echo "----------------------------------------------------------------"

        # Update Config for this run
        # We ensure we point to the Stage 1 weights and set LoRA params
        python utils/update_config.py --config_path "$CONFIG_FILE" \
            --key use_lora --value True \
            --key lora_trained --value False \
            --key max_train_steps --value 5000 \
            --key val_freq --value 300 \
            --key projection_path --value "$STAGE1_EXP_DIR/best_project.pt" \
            --key tokenizer_path --value "$STAGE1_EXP_DIR/tokenizer" \
            --key llm_path --value "$STAGE1_EXP_DIR/best_llm_weights.pt" \
            --key exps_path --value exps \
            --key freeze_projection --value False \
            --key modules_to_save --value "[embed_tokens]" \
            --key projection_lr --value 0.0002 \
            --key llm_lr --value 0.0002 \
            --key warmup_steps --value 0.1 \
            --key r --value 128 \
            --key lora_alpha --value 256 \
            --key lora_dropout --value "$LORA_DROPOUT_VAL" \
            --key llm_gradient_accumulation_steps --value "$LLM_GRADIENT_ACCUM_STEPS_VAL"

        # Set Experiment ID
        export EXP_ID="lora_grid"
        
        echo "Running train_llm.py..."
        python train_llm.py

        # Find output dir
        LATEST_EXP_DIR=$(ls -td exps/*_$EXP_ID | head -1)
        echo "Experiment completed. Output: $LATEST_EXP_DIR"

        # Evaluate
        if [ -f "$LATEST_EXP_DIR/test_preds.json" ]; then
            echo "Evaluating results..."
            python evaluate_llm.py --test_preds_path "$LATEST_EXP_DIR/test_preds.json" > "$LATEST_EXP_DIR/evaluation_results.txt"
            cat "$LATEST_EXP_DIR/evaluation_results.txt"
        else
            echo "Warning: test_preds.json not found."
        fi
        
        echo "Finished run for r=$R, alpha=$ALPHA"
        echo ""
    done
done

echo "LoRA Grid Search Completed."
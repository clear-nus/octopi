#!/bin/bash

export HTTP_PROXY=http://127.0.0.1:1087
export HTTPS_PROXY=http://127.0.0.1:1087

# Array of seeds to test
SEEDS=(0)

# Grid Search Parameters
RANKS=(128)
LRS=(0.0002 0.00005 0.00002)

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
    python src/utils/process_dataset.py --dataset_path "$DATASET_PATH" --output_path "$DATA_DIR" --seed "$SEED"
    python src/utils/generate_qa.py --data_path "$DATA_DIR" --seed "$SEED"

    # 1.5 Train Encoder
    CLIP_CONFIG_FILE="configs/train_clip_config.yaml"
    echo "Training Encoder..."
    python src/utils/update_config.py --config_path "$CLIP_CONFIG_FILE" \
        --key data_dir --value "$DATA_DIR" \
        --key seed --value "$SEED"
    
    export EXP_ID="clip_seed_${SEED}"
    # python src/train_clip.py
    
    # Find encoder output
    LATEST_CLIP_DIR=$(ls -td exps/*_$EXP_ID | head -1)
    ENCODER_PATH="$LATEST_CLIP_DIR/encoder.pt"
    echo "Encoder training completed. Encoder at: $ENCODER_PATH"

    # 2. Run Training (Stage 1)
    # Reset Stage 1 config
    python src/utils/update_config.py --config_path "$CONFIG_FILE" \
        --key seed --value "$SEED" \
        --key encoder_path --value "$ENCODER_PATH" \
        --key use_lora --value False \
        --key max_train_steps --value 8000 \
        --key val_freq --value 800 \
        --key projection_path --value null \
        --key tokenizer_path --value null \
        --key llm_path --value null \
        --key exps_path --value exps \
        --key freeze_projection --value False \
        --key modules_to_save --value "[embed_tokens]" \
        --key projection_lr --value 0.0002 \
        --key llm_lr --value 0.0002 \
        --key warmup_steps --value 10 \
        --key llm_gradient_accumulation_steps --value 16

    # Pass seed as experiment identifier
    export EXP_ID="full_pipeline_${SEED}"
    
    echo "Running train_llm.py (Stage 1)..."
    python src/train_llm.py
    
    # Find output dir of Stage 1
    LATEST_EXP_DIR=$(ls -td exps/*_full_pipeline_$SEED | head -1)
    echo "Stage 1 completed. Output: $LATEST_EXP_DIR"

    # 4. Run LoRA Finetuning (Stage 2 Grid Search)
    echo "Setting up LoRA finetuning grid search..."
    
    for R in "${RANKS[@]}"; do
        for LR in "${LRS[@]}"; do
            ALPHA=$(( R * 2 ))
            echo ""
            echo "================================================================"
            echo "Running LoRA finetuning grid point: r=${R}, alpha=${ALPHA}, lr=${LR}"
            echo "================================================================"
            
            # Update config for LoRA
            python src/utils/update_config.py --config_path "$CONFIG_FILE" \
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
                --key projection_lr --value "$LR" \
                --key llm_lr --value "$LR" \
                --key warmup_steps --value 10 \
                --key r --value "$R" \
                --key lora_alpha --value "$ALPHA" \
                --key lora_dropout --value 0.05 \
                --key target_modules --value "[q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]" \
                --key llm_gradient_accumulation_steps --value 16
            
            export EXP_ID="full_pipeline_${SEED}_lora_r${R}_lr${LR}"
            echo "Running train_llm.py (Stage 2 - LoRA)..."
            python src/train_llm.py

            # Find output dir of Stage 2
            LATEST_LORA_EXP_DIR=$(ls -td exps/*_lora_r${R}_lr${LR} | head -1)
            echo "Stage 2 completed. Output: $LATEST_LORA_EXP_DIR"
        done
    done

    echo "Finished runs for seed $SEED"
    echo ""
done

echo "All LLM training reproducibility runs completed."

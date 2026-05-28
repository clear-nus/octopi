#!/bin/bash
set -e

STAGE1_EXP="exps/2026_05_24_14_17_25_train_llm_train_val_test_vicuna-7b_3200_full_pipeline_0"
ENCODER_PATH="exps/2026_05_24_11_44_45_train_clip_clip_seed_0/encoder.pt"
CONFIG_FILE="configs/train_llm_config.yaml"
SEED=0

RANKS=(32 64 128)
LRS=(0.0001 0.00005 0.00002)

for R in "${RANKS[@]}"; do
    for LR in "${LRS[@]}"; do
        ALPHA=$R
        echo "================================================================"
        echo "Running LoRA: r=$R, alpha=$ALPHA, lr=$LR"
        echo "================================================================"

        python src/utils/update_config.py --config_path "$CONFIG_FILE" \
            --key seed --value "$SEED" \
            --key encoder_path --value "$ENCODER_PATH" \
            --key use_lora --value True \
            --key lora_trained --value False \
            --key max_train_steps --value 3000 \
            --key projection_path --value "$STAGE1_EXP/best_project.pt" \
            --key tokenizer_path --value "$STAGE1_EXP/tokenizer" \
            --key llm_path --value "$STAGE1_EXP/best_llm_weights.pt" \
            --key exps_path --value exps \
            --key freeze_projection --value False \
            --key modules_to_save --value "[embed_tokens]" \
            --key projection_lr --value "$LR" \
            --key llm_lr --value "$LR" \
            --key warmup_steps --value 50 \
            --key val_freq --value 150 \
            --key r --value "$R" \
            --key lora_alpha --value "$ALPHA" \
            --key lora_dropout --value 0.05 \
            --key target_modules --value "[q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]" \
            --key llm_gradient_accumulation_steps --value 16 \
            --key train --value True \
            --key val --value True \
            --key test --value True \
            --key tta_passes --value 1

        export EXP_ID="full_pipeline_${SEED}_lora_r${R}_lr${LR}"
        echo "Running train_llm.py (Stage 2 - LoRA)..."
        python src/train_llm.py

        echo "Finished: r=$R, lr=$LR"
        echo ""
    done
done

echo "Grid search complete!"

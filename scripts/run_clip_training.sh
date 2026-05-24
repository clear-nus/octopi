#!/bin/bash

set -euo pipefail

# Array of seeds to test
SEEDS=(0)

# Path to config file
CLIP_CONFIG_FILE="configs/train_clip_config.yaml"
DATASET_PATH="dataset"
DATA_DIR="data"

for SEED in "${SEEDS[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Running CLIP training experiment with seed: $SEED"
    echo "----------------------------------------------------------------"

    # 1. Generate Data
    echo "Cleaning up data directory..."
    rm -rf "$DATA_DIR"
    echo "Generating data for seed $SEED in $DATA_DIR..."
    python src/utils/process_dataset.py --dataset_path "$DATASET_PATH" --output_path "$DATA_DIR" --seed "$SEED"
    python src/utils/generate_qa.py --data_path "$DATA_DIR" --seed "$SEED"

    # 2. Train Encoder
    echo "Training Encoder..."
    python src/utils/update_config.py --config_path "$CLIP_CONFIG_FILE" \
        --key data_dir --value "$DATA_DIR" \
        --key seed --value "$SEED"

    export EXP_ID="clip_seed_${SEED}"
    python src/train_clip.py

    # Verify output
    LATEST_CLIP_DIR=$(find exps -maxdepth 1 -type d -name "*_clip_seed_${SEED}" | sort | tail -1)
    if [ -z "$LATEST_CLIP_DIR" ] || [ ! -f "$LATEST_CLIP_DIR/encoder.pt" ]; then
        echo "ERROR: Could not find encoder checkpoint after training"
        exit 1
    fi
    echo "Encoder saved to: $LATEST_CLIP_DIR/encoder.pt"

    echo "----------------------------------------------------------------"
    echo "Finished CLIP training for seed: $SEED"
    echo "----------------------------------------------------------------"
done

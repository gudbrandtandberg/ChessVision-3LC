#!/bin/bash

# Set common parameters
BASE_LR=0.0000001
EPOCHS=20
BATCH_SIZE=2
SWEEP_ID=1

TRAIN_TABLE_NAME="initial"
VAL_TABLE_NAME="initial"

source .venv/Scripts/activate

# Function to run a training job
run_training() {
    local lr=$1
    local use_weights=$2
    local threshold=$3
    local seed=42  # Fixed seed for deterministic training
    
    # Construct description from parameters
    local weights_desc=$([ "$use_weights" = true ] && echo "with_weights" || echo "no_weights")
    local description="lr_${lr}_${weights_desc}_thresh_${threshold}"
    
    echo "Starting training with:"
    echo "  Learning rate: $lr"
    echo "  Sample weights: $use_weights"
    echo "  Threshold: $threshold"
    echo "  Description: $description"
    
    # Build weights flag
    local weights_flag=$([ "$use_weights" = true ] && echo "--use-sample-weights" || echo "")
    
    python scripts/train/train_unet.py \
        --learning-rate "$lr" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --threshold "$threshold" \
        --seed "$seed" \
        --deterministic \
        --amp \
        $weights_flag \
        --run-description "$description" \
        --sweep-id "$SWEEP_ID" \
        --train-table "$TRAIN_TABLE_NAME" \
        --val-table "$VAL_TABLE_NAME" \

    echo "Training completed for $description"
    echo "----------------------------------------"
}

# Parameters to sweep
LR_MULTIPLIERS=(1.0 10.0 100.0 1000.0)
THRESHOLDS=(0.3 0.5 0.7)

# Run full sweep over learning rates, sample weights, and thresholds
for multiplier in "${LR_MULTIPLIERS[@]}"; do
    lr=$(awk "BEGIN {print $BASE_LR * $multiplier}")
    
    for threshold in "${THRESHOLDS[@]}"; do
        # Without sample weights
        run_training "$lr" false "$threshold"
        
        # With sample weights
        run_training "$lr" true "$threshold"
    done
done

echo "All training jobs completed!"
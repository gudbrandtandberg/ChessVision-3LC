#!/bin/bash

# Set common parameters
BASE_LR=0.0000001
EPOCHS=20
BATCH_SIZE=2
PROJECT_NAME="chessvision-segmentation"
THRESHOLD=0.3

source .venv-dev/Scripts/activate

# Function to run a training job
run_training() {
    local lr=$1
    local use_weights=$2
    local seed=42  # Fixed seed for deterministic training
    
    # Construct description from parameters
    local weights_desc=$([ "$use_weights" = true ] && echo "with_weights" || echo "no_weights")
    local description="lr_${lr}_${weights_desc}"
    
    echo "Starting training with:"
    echo "  Learning rate: $lr"
    echo "  Sample weights: $use_weights"
    echo "  Description: $description"
    
    # Build weights flag
    local weights_flag=$([ "$use_weights" = true ] && echo "--use-sample-weights" || echo "")
    
    python chessvision/board_extraction/train_unet.py \
        --learning-rate "$lr" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --project-name "$PROJECT_NAME" \
        --threshold "$THRESHOLD" \
        --seed "$seed" \
        --deterministic \
        --amp \
        --run-tests \
        $weights_flag \
        --run-description "$description"
    
    echo "Training completed for $description"
    echo "----------------------------------------"
}

# Learning rates to try (multipliers of BASE_LR)
LR_MULTIPLIERS=(1.0 10.0 100.0 1000.0)

# Run sweep over learning rates and sample weights
for multiplier in "${LR_MULTIPLIERS[@]}"; do
    lr=$(awk "BEGIN {print $BASE_LR * $multiplier}")
    
    # Without sample weights
    run_training "$lr" false
    
    # With sample weights
    run_training "$lr" true
done

echo "All training jobs completed!"
#!/bin/bash

# ======================
# Configuration Section
# ======================

# Model parameters
LEARNING_RATE=0.000001
EPOCHS=20
BATCH_SIZE=2
THRESHOLD=0.5

# Training options
USE_SAMPLE_WEIGHTS=true
USE_AMP=true
USE_DETERMINISTIC=true
AUGMENT=true
SKIP_EVAL=false

# Project settings
TRAIN_TABLE_NAME="initial"
VAL_TABLE_NAME="initial"
RUN_DESCRIPTION=""
SWEEP_ID=0

# ======================
# Activate Environment
# ======================
source .venv/Scripts/activate

# ======================
# Run Training
# ======================

# Build sample weights flag
WEIGHTS_FLAG=$([ "$USE_SAMPLE_WEIGHTS" = true ] && echo "--use-sample-weights" || echo "")

# Build other flags
AMP_FLAG=$([ "$USE_AMP" = true ] && echo "--amp" || echo "")
DETERMINISTIC_FLAG=$([ "$USE_DETERMINISTIC" = true ] && echo "--deterministic" || echo "")
AUGMENT_FLAG=$([ "$AUGMENT" = true ] && echo "--augment" || echo "")
SKIP_EVAL_FLAG=$([ "$SKIP_EVAL" = true ] && echo "--skip-eval" || echo "")

echo "Starting training with:"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Threshold: $THRESHOLD"
echo "  Sample weights: $USE_SAMPLE_WEIGHTS"
echo "  Mixed precision: $USE_AMP"
echo "  Deterministic: $USE_DETERMINISTIC"
echo "  Train table: $TRAIN_TABLE_NAME"
echo "  Val table: $VAL_TABLE_NAME"

python scripts/train/train_unet.py \
    --learning-rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --threshold "$THRESHOLD" \
    --seed 42 \
    $WEIGHTS_FLAG \
    $AMP_FLAG \
    $DETERMINISTIC_FLAG \
    --run-description "$RUN_DESCRIPTION" \
    --sweep-id "$SWEEP_ID" \
    --train-table "$TRAIN_TABLE_NAME" \
    --val-table "$VAL_TABLE_NAME" \
    $AUGMENT_FLAG \
    $SKIP_EVAL_FLAG

echo "Training completed!"

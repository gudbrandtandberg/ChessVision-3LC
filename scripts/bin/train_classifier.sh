#!/bin/bash

# ======================
# Configuration Section
# ======================

# Model parameters
LEARNING_RATE=0.001  # Higher LR since we're using warmup
EPOCHS=20  # More epochs for classification
BATCH_SIZE=64  # Larger batch size for classification
PATIENCE=10  # More patience for early stopping
COLLECTION_FREQ=5

# Training options
USE_SAMPLE_WEIGHTS=true
USE_DETERMINISTIC=true
RUN_TESTS=true

# Project settings
TRAIN_TABLE_NAME="initial"
VAL_TABLE_NAME="initial"
RUN_DESCRIPTION="Training with improved augmentation and warmup"
SWEEP_ID=0

# ======================
# Activate Environment
# ======================
source .venv/Scripts/activate

# ======================
# Run Training
# ======================

# Build flags
WEIGHTS_FLAG=$([ "$USE_SAMPLE_WEIGHTS" = true ] && echo "--use-sample-weights" || echo "")
DETERMINISTIC_FLAG=$([ "$USE_DETERMINISTIC" = true ] && echo "--deterministic" || echo "")
TEST_FLAG=$([ "$RUN_TESTS" = true ] && echo "--run-tests" || echo "")

echo "Starting classifier training with:"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Patience: $PATIENCE"
echo "  Collection frequency: $COLLECTION_FREQ"
echo "  Sample weights: $USE_SAMPLE_WEIGHTS"
echo "  Deterministic: $USE_DETERMINISTIC"
echo "  Run tests: $RUN_TESTS"
echo "  Train table: $TRAIN_TABLE_NAME"
echo "  Val table: $VAL_TABLE_NAME"

python -m scripts.train.train_classifier \
    --learning-rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    --collection-frequency "$COLLECTION_FREQ" \
    --seed 42 \
    $WEIGHTS_FLAG \
    $DETERMINISTIC_FLAG \
    $TEST_FLAG \
    --run-description "$RUN_DESCRIPTION" \
    --sweep-id "$SWEEP_ID" \
    --train-table "$TRAIN_TABLE_NAME" \
    --val-table "$VAL_TABLE_NAME"

echo "Training completed!"

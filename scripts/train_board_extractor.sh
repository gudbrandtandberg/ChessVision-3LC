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
RUN_TESTS=true

# Project settings
PROJECT_NAME="chessvision-segmentation"
TRAIN_DATASET_NAME="chessboard-segmentation-train"
VAL_DATASET_NAME="chessboard-segmentation-val"
TRAIN_TABLE_NAME="table"
VAL_TABLE_NAME="table"
RUN_DESCRIPTION="TEST_DELETE_ME"
SWEEP_ID=0

# ======================
# Activate Environment
# ======================
source .venv-dev/Scripts/activate

# ======================
# Run Training
# ======================

# Build sample weights flag
WEIGHTS_FLAG=$([ "$USE_SAMPLE_WEIGHTS" = true ] && echo "--use-sample-weights" || echo "")

# Build other flags
AMP_FLAG=$([ "$USE_AMP" = true ] && echo "--amp" || echo "")
DETERMINISTIC_FLAG=$([ "$USE_DETERMINISTIC" = true ] && echo "--deterministic" || echo "")
TEST_FLAG=$([ "$RUN_TESTS" = true ] && echo "--run-tests" || echo "")

echo "Starting training with:"
echo "  Learning rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Threshold: $THRESHOLD"
echo "  Sample weights: $USE_SAMPLE_WEIGHTS"
echo "  Mixed precision: $USE_AMP"
echo "  Deterministic: $USE_DETERMINISTIC"
echo "  Run tests: $RUN_TESTS"

python chessvision/board_extraction/train_unet.py \
    --learning-rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --project-name "$PROJECT_NAME" \
    --threshold "$THRESHOLD" \
    --seed 42 \
    $WEIGHTS_FLAG \
    $AMP_FLAG \
    $DETERMINISTIC_FLAG \
    $TEST_FLAG \
    --run-description "$RUN_DESCRIPTION" \
    --sweep-id "$SWEEP_ID" \
    --train-table "$TRAIN_TABLE_NAME" \
    --val-table "$VAL_TABLE_NAME" \
    --train-dataset "$TRAIN_DATASET_NAME" \
    --val-dataset "$VAL_DATASET_NAME"

echo "Training completed!"

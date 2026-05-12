#!/bin/bash

# Simple wrapper for train_yolo_classifier.py
# Edit the arguments below as needed

uv run python scripts/train/train_yolo_classifier.py \
    --model "yolov8n-cls.pt" \
    --epochs 5 \
    --patience 5 \
    --batch-size -1 \
    --run-name "Original Model" \
    --run-description "Original model with no modifications" \
    --train-table-name "initial" \
    --val-table-name "initial" \
    --use-sample-weights \
    # --collection-frequency 1 \
    # --skip-eval

#true-travel-distance-zero
#pacmap-travel-distance-zero
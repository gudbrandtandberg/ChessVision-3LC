#!/bin/bash

# Simple wrapper for train_yolo_classifier.py
# Edit the arguments below as needed

python scripts/train/train_yolo_classifier.py \
    --model "yolov8m-cls.pt" \
    --epochs 5 \
    --patience 5 \
    --batch-size -1 \
    --run-name "" \
    --run-description "" \
    --use-sample-weights \
    --train-table-name "initial" \
    --val-table-name "initial" \
    --collection-frequency 1 \
    --skip-eval

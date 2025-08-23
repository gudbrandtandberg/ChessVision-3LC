#!/bin/bash

# Simple wrapper for train_classifier.py
# Edit the arguments below as needed

python scripts/train/train_classifier.py \
    --learning-rate 0.001 \
    --epochs 10 \
    --batch-size 32 \
    --patience 5 \
    --collection-frequency 5 \
    --seed 42 \
    --use-sample-weights \
    --deterministic \
    --run-name "" \
    --run-description "" \
    --sweep-id 0 \
    --train-table "initial" \
    --val-table "initial" \
    --skip-eval

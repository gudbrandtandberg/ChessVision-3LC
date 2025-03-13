#!/bin/bash

# Simple wrapper for train_unet.py
# Edit the arguments below as needed

python scripts/train/train_unet.py \
    --learning-rate 0.000001 \
    --epochs 5 \
    --batch-size 2 \
    --threshold 0.5 \
    --seed 42 \
    --use-sample-weights \
    --amp \
    --deterministic \
    --run-description "" \
    --run-name "" \
    --sweep-id 0 \
    --collection-frequency 5 \
    --patience 5 \
    --train-table "initial" \
    --val-table "initial" \
    --augment \
    --skip-eval

#!/bin/bash

# Simple wrapper for evaluate.py
# Edit the arguments below as needed

# --board-extractor-model-id "" mean that UNet is used

python scripts/eval/evaluate.py \
    --image-folder "data/test/initial/raw" \
    --threshold 0.5 \
    --project-name "chessvision-testing" \
    --run-name "" \
    --run-description "" \
    --board-extractor-weights "weights/best_extractor.pth" \
    --classifier-weights "weights/best_yolo_classifier.pt" \
    --classifier-model-id "yolo" \
    --include-metrics-table

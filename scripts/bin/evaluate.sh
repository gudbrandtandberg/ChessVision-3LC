#!/bin/bash

# Simple wrapper for evaluate.py
# Edit the arguments below as needed

python scripts/eval/evaluate.py \
    --image-folder "data/test/intitial/raw" \
    --threshold 0.5 \
    --project-name "chessvision-testing" \
    --run-name "" \
    --run-description "" \
    --board-extractor-weights "" \
    --classifier-weights "" \
    --classifier-model-id "yolo" \
#    --table-name "merged-2024-11-04-2024-11-04"


# Simple wrapper for train_yolo_segmentation_model.py
# Edit the arguments below as needed

python scripts/train/train_yolo_segmentation_model.py \
    --model "yolo11n-seg.pt" \
    --epochs 15 \
    --patience 5 \
    --batch-size -1 \
    --run-name "" \
    --run-description "Try yolo11n"

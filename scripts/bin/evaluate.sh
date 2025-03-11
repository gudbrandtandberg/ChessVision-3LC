#!/bin/bash

# Default values
IMAGE_FOLDER="data/test/raw"
TRUTH_FOLDER="data/test/ground_truth"
THRESHOLD=0.5
PROJECT_NAME="chessvision-testing"
RUN_NAME=""
RUN_DESCRIPTION=""
BOARD_EXTRACTOR_WEIGHTS=""
CLASSIFIER_WEIGHTS=""
CLASSIFIER_MODEL="yolo"

# Help text
show_help() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo
    echo "Evaluate ChessVision model on test dataset"
    echo
    echo "Options:"
    echo "  -i, --image-folder PATH      Path to test images folder (default: $IMAGE_FOLDER)"
    echo "  -t, --truth-folder PATH      Path to ground truth folder (default: $TRUTH_FOLDER)"
    echo "  -T, --threshold VALUE        Board detection threshold (default: $THRESHOLD)"
    echo "  -p, --project-name NAME      Project name for logging (default: $PROJECT_NAME)"
    echo "  -n, --run-name NAME          Run name for logging (default: auto-generated)"
    echo "  -d, --description TEXT       Run description"
    echo "  -b, --board-weights PATH     Path to board extractor weights"
    echo "  -c, --classifier-weights PATH Path to classifier weights"
    echo "  -m, --model NAME            Classifier model ID (default: $CLASSIFIER_MODEL)"
    echo "  -h, --help                   Show this help message"
    echo
    echo "Example:"
    echo "  $(basename "$0") -i custom/test/images -t custom/test/labels -m yolo"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image-folder)
            IMAGE_FOLDER="$2"
            shift 2
            ;;
        -t|--truth-folder)
            TRUTH_FOLDER="$2"
            shift 2
            ;;
        -T|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -p|--project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -n|--run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        -d|--description)
            RUN_DESCRIPTION="$2"
            shift 2
            ;;
        -b|--board-weights)
            BOARD_EXTRACTOR_WEIGHTS="$2"
            shift 2
            ;;
        -c|--classifier-weights)
            CLASSIFIER_WEIGHTS="$2"
            shift 2
            ;;
        -m|--model)
            CLASSIFIER_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required paths
if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "Error: Image folder '$IMAGE_FOLDER' does not exist"
    exit 1
fi

if [ ! -d "$TRUTH_FOLDER" ]; then
    echo "Error: Truth folder '$TRUTH_FOLDER' does not exist"
    exit 1
fi

# Validate board extractor weights if specified
if [ -n "$BOARD_EXTRACTOR_WEIGHTS" ] && [ ! -f "$BOARD_EXTRACTOR_WEIGHTS" ]; then
    echo "Error: Board extractor weights '$BOARD_EXTRACTOR_WEIGHTS' not found"
    exit 1
fi

# Validate classifier weights if specified
if [ -n "$CLASSIFIER_WEIGHTS" ] && [ ! -f "$CLASSIFIER_WEIGHTS" ]; then
    echo "Error: Classifier weights '$CLASSIFIER_WEIGHTS' not found"
    exit 1
fi

# Build command
CMD="python -m scripts.eval.evaluate"
CMD="$CMD --image-folder '$IMAGE_FOLDER'"
CMD="$CMD --truth-folder '$TRUTH_FOLDER'"
CMD="$CMD --threshold $THRESHOLD"
CMD="$CMD --project-name '$PROJECT_NAME'"

if [ -n "$RUN_NAME" ]; then
    CMD="$CMD --run-name '$RUN_NAME'"
fi

if [ -n "$RUN_DESCRIPTION" ]; then
    CMD="$CMD --run-description '$RUN_DESCRIPTION'"
fi

if [ -n "$BOARD_EXTRACTOR_WEIGHTS" ]; then
    CMD="$CMD --board-extractor-weights '$BOARD_EXTRACTOR_WEIGHTS'"
fi

if [ -n "$CLASSIFIER_WEIGHTS" ]; then
    CMD="$CMD --classifier-weights '$CLASSIFIER_WEIGHTS'"
fi

if [ -n "$CLASSIFIER_MODEL" ]; then
    CMD="$CMD --classifier-model-id '$CLASSIFIER_MODEL'"
fi

# Run evaluation
echo "Starting evaluation..."
echo "Command: $CMD"
eval "$CMD"

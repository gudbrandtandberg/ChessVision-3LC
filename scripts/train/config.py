"""Training configuration and constants."""

from pathlib import Path

import tlc

from chessvision import constants

# Project names
BOARD_EXTRACTION_PROJECT = "chessvision-segmentation"
PIECE_CLASSIFICATION_PROJECT = "chessvision-classification"
YOLO_CLASSIFICATION_PROJECT = "chessvision-yolo-classification"

# Dataset paths
BOARD_EXTRACTION_ROOT = Path(constants.DATA_ROOT) / "board_extraction"
PIECE_CLASSIFICATION_ROOT = Path(constants.DATA_ROOT) / "squares"

# Board extraction config
BOARD_EXTRACTION_PATHS = {
    "images": BOARD_EXTRACTION_ROOT / "images",
    "masks": BOARD_EXTRACTION_ROOT / "masks",
}
BOARD_EXTRACTION_DATASETS = {
    "train": "chessboard-segmentation-train",
    "val": "chessboard-segmentation-val",
}
VAL_SPLIT_PERCENT = 0.1

# Piece classification config
PIECE_CLASSIFICATION_PATHS = {
    "train": PIECE_CLASSIFICATION_ROOT / "training",
    "val": PIECE_CLASSIFICATION_ROOT / "validation",
}
PIECE_CLASSIFICATION_DATASETS = {
    "train": "chesspieces-train",
    "val": "chesspieces-val",
}

# Initial table names
INITIAL_TABLE_NAME = "initial"

# Register TLC aliases
tlc.register_url_alias(
    "CHESSVISION_SEGMENTATION_DATA_ROOT",
    BOARD_EXTRACTION_ROOT,
)
tlc.register_url_alias(
    "CHESSVISION_SEGMENTATION_PROJECT_ROOT",
    f"{tlc.Configuration.instance().project_root_url}/{BOARD_EXTRACTION_PROJECT}",
)
tlc.register_url_alias(
    "CHESSPIECES_DATASET_ROOT",
    PIECE_CLASSIFICATION_ROOT,
)
tlc.register_url_alias(
    "CHESSPIECES_PROJECT_ROOT",
    f"{tlc.Configuration.instance().project_root_url}/{PIECE_CLASSIFICATION_PROJECT}",
)

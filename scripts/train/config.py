"""Training configuration and constants."""

import tlc

from chessvision import constants


def _register_url_alias(token: str, path: object) -> None:
    """Register a URL alias across tlc versions.

    `tlc.register_url_alias` was top-level in tlc 3.0 and moved to `tlc.url` in 3.1.
    """
    if hasattr(tlc, "register_url_alias"):
        tlc.register_url_alias(token, path)
    else:
        from tlc.url import register_url_alias

        register_url_alias(token, path)


def _project_root_url() -> str:
    """Project root URL across tlc versions.

    `tlc.Configuration.instance()` (3.0) became the live `tlc.config` singleton in 3.1.
    """
    if hasattr(tlc, "Configuration"):
        return tlc.Configuration.instance().project_root_url
    return tlc.config.project_root_url

# Project names
BOARD_EXTRACTION_PROJECT = "chessvision-segmentation"
PIECE_CLASSIFICATION_PROJECT = "chessvision-classification"
YOLO_CLASSIFICATION_PROJECT = "chessvision-classification"

# Dataset paths
BOARD_EXTRACTION_ROOT = constants.DATA_ROOT / "board_extraction"
PIECE_CLASSIFICATION_ROOT = constants.DATA_ROOT / "squares"

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
_register_url_alias(
    "CHESSVISION_SEGMENTATION_DATA_ROOT",
    BOARD_EXTRACTION_ROOT,
)
_register_url_alias(
    "CHESSVISION_SEGMENTATION_PROJECT_ROOT",
    f"{_project_root_url()}/{BOARD_EXTRACTION_PROJECT}",
)
_register_url_alias(
    "CHESSPIECES_DATASET_ROOT",
    PIECE_CLASSIFICATION_ROOT,
)
_register_url_alias(
    "CHESSPIECES_PROJECT_ROOT",
    f"{_project_root_url()}/{PIECE_CLASSIFICATION_PROJECT}",
)

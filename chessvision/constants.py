"""ChessVision constants and configuration."""

import os
from pathlib import Path

# Root paths
CVROOT = os.getenv("CVROOT", Path(__file__).parent.parent.as_posix())
DATA_ROOT = Path(CVROOT) / "data"

# Resource paths
BLACK_BOARD_PATH = (DATA_ROOT / "board_extraction" / "black_board.png").as_posix()
BLACK_SQUARE_PATH = (DATA_ROOT / "squares" / "black_square.png").as_posix()

# Model configuration
NUM_CLASSES = 13

# Image sizes
INPUT_SIZE = (256, 256)
BOARD_SIZE = (512, 512)
PIECE_SIZE = (64, 64)

# Label mappings
LABEL_NAMES = ["B", "K", "N", "P", "Q", "R", "b", "k", "n", "p", "q", "r", "f"]
LABEL_INDICES = {label: idx for idx, label in enumerate(LABEL_NAMES)}
LABEL_DESCRIPTIONS = [
    "White Bishop",
    "White King",
    "White Knight",
    "White Pawn",
    "White Queen",
    "White Rook",
    "Black Bishop",
    "Black King",
    "Black Knight",
    "Black Pawn",
    "Black Queen",
    "Black Rook",
    "Empty Square",
    "Unknown",
]

# Segmentation mapping
SEGMENTATION_MAP = {0: "background", 255: "chessboard"}

# Model weights paths
WEIGHTS_DIR = Path(CVROOT) / "weights"
BEST_CLASSIFIER_WEIGHTS = str(WEIGHTS_DIR / "best_classifier.pth")
BEST_EXTRACTOR_WEIGHTS = str(WEIGHTS_DIR / "best_extractor.pth")
BEST_YOLO_CLASSIFIER = str(WEIGHTS_DIR / "best_yolo_classifier.pt")

# Chess board constants
DARK_SQUARES = {
    "a1",
    "c1",
    "e1",
    "g1",
    "b2",
    "d2",
    "f2",
    "h2",
    "a3",
    "c3",
    "e3",
    "g3",
    "b4",
    "d4",
    "f4",
    "h4",
    "a5",
    "c5",
    "e5",
    "g5",
    "b6",
    "d6",
    "f6",
    "h6",
    "a7",
    "c7",
    "e7",
    "g7",
    "b8",
    "d8",
    "f8",
    "h8",
}

INVALID_PAWN_SQUARES = {
    "a1",
    "b1",
    "c1",
    "d1",
    "e1",
    "f1",
    "g1",
    "h1",
    "a8",
    "b8",
    "c8",
    "d8",
    "e8",
    "f8",
    "g8",
    "h8",
}

# Pre-compute square names for both orientations
SQUARE_NAMES_NORMAL = [
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
]

SQUARE_NAMES_FLIPPED = [
    "h1", "g1", "f1", "e1", "d1", "c1", "b1", "a1",
    "h2", "g2", "f2", "e2", "d2", "c2", "b2", "a2",
    "h3", "g3", "f3", "e3", "d3", "c3", "b3", "a3",
    "h4", "g4", "f4", "e4", "d4", "c4", "b4", "a4",
    "h5", "g5", "f5", "e5", "d5", "c5", "b5", "a5",
    "h6", "g6", "f6", "e6", "d6", "c6", "b6", "a6",
    "h7", "g7", "f7", "e7", "d7", "c7", "b7", "a7",
    "h8", "g8", "f8", "e8", "d8", "c8", "b8", "a8",
]

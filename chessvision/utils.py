import os
from pathlib import Path

import numpy as np
from PIL import Image

CVROOT = os.getenv("CVROOT", Path(__file__).parent.parent.as_posix())

DATA_ROOT = (Path(CVROOT) / "data").as_posix()

INPUT_SIZE = (256, 256)
BOARD_SIZE = (512, 512)
PIECE_SIZE = (64, 64)

labels = {
    "B": 0,
    "K": 1,
    "N": 2,
    "P": 3,
    "Q": 4,
    "R": 5,
    "b": 6,
    "k": 7,
    "n": 8,
    "p": 9,
    "q": 10,
    "r": 11,
    "f": 12,
}

label_names = [
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

segmentation_map = {0: "background", 255: "chessboard"}

classifier_weights_dir = (Path(CVROOT) / "weights" / "classifier").as_posix()
extractor_weights_dir = (Path(CVROOT) / "weights" / "extractor").as_posix()

best_classifier_weights = (Path(CVROOT) / "weights" / "best_classifier.pth").as_posix()
best_extractor_weights = (Path(CVROOT) / "weights" / "best_extractor.pth").as_posix()


def listdir_nohidden(path):
    files = []
    for f in os.listdir(path):
        if not f.startswith("."):
            files.append(f)
    return files


def ratio(a, b):
    if a == 0 or b == 0:
        return -1
    return min(a, b) / float(max(a, b))


def get_device():
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


BLACK_BOARD = Image.fromarray(np.zeros(INPUT_SIZE).astype(np.uint8))
BLACK_CROP = Image.fromarray(np.zeros(PIECE_SIZE).astype(np.uint8))

BLACK_BOARD_URL = DATA_ROOT + "/board_extraction/black_board.png"
BLACK_CROP_URL = DATA_ROOT + "/squares/black_square.png"

if not Path(BLACK_BOARD_URL).exists():
    BLACK_BOARD.save(BLACK_BOARD_URL)

if not Path(BLACK_CROP_URL).exists():
    BLACK_CROP.save(BLACK_CROP_URL)

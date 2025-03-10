"""Utility functions for ChessVision."""

from __future__ import annotations

import os

import cv2
import numpy as np
import timm
import torch
from numpy.typing import NDArray

from . import constants


def get_device() -> torch.device:
    """Get the best available device for PyTorch."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_classifier_model(model_id: str = "resnet18") -> torch.nn.Module:
    """Initialize the piece classifier model."""
    return timm.create_model(  # type: ignore[no-any-return]
        model_id,
        num_classes=constants.NUM_CLASSES,
        in_chans=1,
    )


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load a model checkpoint."""
    if device is None:
        device = get_device()

    state_dict = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(state_dict, dict):
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
            metadata = state_dict.get("metadata", {})
        elif "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"])
            metadata = state_dict.get("metadata", {})
        else:
            if "model" in state_dict:
                model_weights = state_dict["model"]
                metadata = state_dict.get("metadata", {})
            else:
                model_weights = state_dict
                metadata = {}
            model.load_state_dict(model_weights)
    else:
        model.load_state_dict(state_dict)
        metadata = {}

    if metadata:
        model.metadata = metadata

    return model


def ratio(a: float, b: float) -> float:
    """Calculate ratio between two numbers."""
    if a == 0 or b == 0:
        return -1
    return min(a, b) / float(max(a, b))


def listdir_nohidden(path: str) -> list[str]:
    """List directory contents, excluding hidden files."""
    return [f for f in os.listdir(path) if not f.startswith(".")]


def create_binary_mask(mask: NDArray[np.float32], threshold: float = 0.5) -> NDArray[np.uint8]:
    """Convert probability mask to binary mask."""
    mask = mask.copy()
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0
    return mask.astype(np.uint8)


def extract_perspective(
    image: NDArray[np.uint8],
    approx: NDArray[np.float32],
    out_size: tuple[int, int],
) -> NDArray[np.uint8]:
    """Extract a perspective-corrected region from an image."""
    w, h = out_size[0], out_size[1]
    dest = np.array(((0, 0), (w, 0), (w, h), (0, h)), np.float32)
    approx = np.array(approx, np.float32)

    coeffs = cv2.getPerspectiveTransform(approx, dest)
    return cv2.warpPerspective(image, coeffs, out_size)


def display_comparison(
    original_img: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    board_img: NDArray[np.uint8],
    fen: str,
    figsize: tuple[int, int] = (20, 5),
) -> None:
    import io

    import cairosvg
    import chess
    import chess.svg
    import matplotlib.pyplot as plt

    _, axes = plt.subplots(1, 4, figsize=figsize)

    # Original image
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Segmentation mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")

    # Extracted board
    axes[2].imshow(board_img, cmap="gray")
    axes[2].set_title("Extracted Board")
    axes[2].axis("off")

    # Chess position
    if fen:
        board = chess.Board(fen)
        svg_board = chess.svg.board(board, size=300)
        axes[3].axis("off")
        axes[3].set_title("Detected Position")

        # Convert SVG to a format matplotlib can display
        svg_img = cairosvg.svg2png(bytestring=svg_board.encode())
        chess_img = plt.imread(io.BytesIO(svg_img))
        axes[3].imshow(chess_img)
    else:
        axes[3].text(0.5, 0.5, "No valid FEN detected", horizontalalignment="center", verticalalignment="center")
        axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def load_yolo_model(model_weights: str) -> torch.nn.Module:
    """Load a YOLO model for piece classification.

    Args:
        model_weights: Path to YOLO model weights

    Returns:
        Wrapped YOLO model that implements the classifier interface

    Raises:
        ImportError: If ultralytics is not installed
    """
    try:
        from ultralytics.utils.tlc import TLCYOLO
    except ImportError:
        logger.warning(
            "YOLO model requires ultralytics package. Please install with 'pip install git+https://github.com/3lc-ai/ultralytics.git'."
        )
        raise

    class YOLOModelWrapper:
        """Wrapper to make YOLO model behave like a classifier."""

        def __init__(self, model: TLCYOLO):
            self.model = model

        def __call__(self, img: torch.Tensor) -> torch.Tensor:
            """Forward pass that returns probabilities for each class."""
            res = self.model(img.repeat((1, 3, 1, 1)), verbose=False)
            return torch.vstack([r.probs.data for r in res])

        def eval(self) -> None:
            """Set the model to evaluation mode."""
            self.model.eval()

        def train(self) -> None:
            """Set the model to training mode."""
            self.model.train()

        def to(self, device: torch.device) -> None:
            """Move the model to a specific device."""
            self.model.to(device)

    return YOLOModelWrapper(TLCYOLO(model_weights))

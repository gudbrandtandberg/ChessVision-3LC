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


def create_binary_mask(mask: NDArray[np.uint8], threshold: float = 0.5) -> NDArray[np.uint8]:
    """Convert probability mask to binary mask."""
    mask = mask.copy()
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0
    return mask.astype(np.uint8)


def extract_perspective(
    image: NDArray[np.uint8],
    approx: NDArray[np.uint8],
    out_size: tuple[int, int],
) -> NDArray[np.uint8]:
    """Extract a perspective-corrected region from an image."""
    w, h = out_size[0], out_size[1]
    dest = np.array(((0, 0), (w, 0), (w, h), (0, h)), np.float32)
    approx = np.array(approx, np.float32)

    coeffs = cv2.getPerspectiveTransform(approx, dest)
    return cv2.warpPerspective(image, coeffs, out_size)

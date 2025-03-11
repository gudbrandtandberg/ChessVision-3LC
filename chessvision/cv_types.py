"""Type definitions for ChessVision."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BoardExtractionResult:
    """Results from board extraction stage."""

    board_image: Optional[np.ndarray]  # The extracted board image, or None if no board found
    logits: np.ndarray  # Raw model output
    probabilities: np.ndarray  # Model probabilities
    binary_mask: np.ndarray  # Thresholded mask
    quadrangle: Optional[np.ndarray]  # The detected quadrangle, or None if no board found


@dataclass
class PositionResult:
    """Results from position classification stage."""

    fen: str  # FEN string representation of the position
    predictions: np.ndarray  # Raw model predictions for each square
    squares: np.ndarray  # Individual square images
    square_names: list[str]  # Chess coordinates for each square
    confidence_scores: dict[str, float]  # Confidence scores per square


@dataclass
class ChessVisionResult:
    """Complete results from image processing."""

    board_extraction: BoardExtractionResult
    position: Optional[PositionResult]  # None if board extraction failed
    processing_time: float  # Total processing time in seconds

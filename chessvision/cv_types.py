"""Type definitions for ChessVision."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class ValidationFix:
    """Record of a validation rule fix applied to a position."""

    square_name: str  # Chess coordinate (e.g. "e4")
    original_piece: str  # Original piece symbol (e.g. "P")
    corrected_piece: str  # Corrected piece symbol (e.g. "Q")
    rule_name: str  # Name of the validation rule that triggered the fix


@dataclass
class BoardExtractionResult:
    """Results from board extraction stage."""

    probabilities: NDArray[np.float32]
    binary_mask: NDArray[np.uint8]  # Thresholded mask
    quadrangle: Optional[NDArray[np.float32]]  # The detected quadrangle, or None if no board found
    board_image: Optional[NDArray[np.uint8]]  # The extracted board image, or None if no board found


@dataclass
class PositionResult:
    """Results from position classification stage including validation."""

    fen: str  # Final FEN after validation
    original_fen: str  # FEN before validation
    model_probabilities: NDArray[np.float32]  # Raw model probabilities (64, 13)
    squares: NDArray[np.uint8]  # Individual square images (64, 64, 64, 1)
    square_names: list[str]  # Chess coordinates for each square
    validation_fixes: list[ValidationFix]  # List of validation fixes applied


@dataclass
class ChessVisionResult:
    """Complete results from image processing."""

    board_extraction: BoardExtractionResult
    position: Optional[PositionResult]  # None if board extraction failed
    processing_time: float  # Total processing time in seconds


@dataclass
class ValidationMetrics:
    """Metrics comparing position accuracy before and after validation."""

    accuracy_before: float  # Accuracy before validation rules applied
    accuracy_after: float  # Accuracy after validation rules applied
    num_fixes: int  # Number of validation fixes applied
    fixes: list[ValidationFix]  # List of specific fixes made

    @property
    def accuracy_delta(self) -> float:
        """Change in accuracy from validation."""
        return self.accuracy_after - self.accuracy_before

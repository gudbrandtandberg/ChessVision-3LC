"""Core ChessVision functionality."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import chess
import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F  # noqa: N812

from .pytorch_unet.unet.unet_model import UNet
from .types import BoardExtractionResult, ChessVisionResult, PositionResult

logger = logging.getLogger(__name__)


class ChessVision:
    """Main class for chess position detection from images."""

    # Root paths
    CVROOT = os.getenv("CVROOT", Path(__file__).parent.parent.as_posix())
    DATA_ROOT = (Path(CVROOT) / "data").as_posix()

    # Resource paths
    BLACK_BOARD_PATH = (Path(DATA_ROOT) / "board_extraction" / "black_board.png").as_posix()
    BLACK_SQUARE_PATH = (Path(DATA_ROOT) / "squares" / "black_square.png").as_posix()

    # Model configuration
    NUM_CLASSES = 13
    MODEL_ID = "resnet18"

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
    CLASSIFIER_WEIGHTS = str(WEIGHTS_DIR / "best_classifier.pth")
    EXTRACTOR_WEIGHTS = str(WEIGHTS_DIR / "best_extractor.pth")

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

    def __init__(
        self,
        board_extractor_weights: str | None = None,
        classifier_weights: str | None = None,
        lazy_load: bool = True,
    ):
        """Initialize ChessVision with optional custom model weights.

        Args:
            board_extractor_weights: Path to board extraction model weights.
                                   If None, uses best available weights.
            classifier_weights: Path to piece classifier model weights.
                              If None, uses best available weights.
            lazy_load: If True, models are loaded only when needed.
                      If False, models are loaded immediately.
        """
        self.device = self.get_device()
        self._board_extractor: torch.nn.Module | None = None
        self._classifier: torch.nn.Module | None = None
        self._board_extractor_weights = board_extractor_weights or self.EXTRACTOR_WEIGHTS
        self._classifier_weights = classifier_weights or self.CLASSIFIER_WEIGHTS

        if not lazy_load:
            self._initialize_board_extractor()
            self._initialize_classifier()

    @staticmethod
    def get_device() -> torch.device:
        """Get the best available device for PyTorch."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _ratio(a: float, b: float) -> float:
        """Calculate ratio between two numbers."""
        if a == 0 or b == 0:
            return -1
        return min(a, b) / float(max(a, b))

    @staticmethod
    def _listdir_nohidden(path: str) -> list[str]:
        """List directory contents, excluding hidden files."""
        return [f for f in os.listdir(path) if not f.startswith(".")]

    @property
    def board_extractor(self) -> torch.nn.Module:
        """Get the board extractor model, initializing if needed."""
        if self._board_extractor is None:
            self._initialize_board_extractor()
        return self._board_extractor

    @property
    def classifier(self) -> torch.nn.Module:
        """Get the classifier model, initializing if needed."""
        if self._classifier is None:
            self._initialize_classifier()
        return self._classifier

    def _initialize_board_extractor(self) -> None:
        """Initialize the board extraction model."""
        logger.info("Initializing board extraction model...")
        self._board_extractor = UNet(n_channels=3, n_classes=1)
        self._board_extractor = self._board_extractor.to(memory_format=torch.channels_last)
        self._board_extractor = self.load_model_checkpoint(
            self._board_extractor,
            self._board_extractor_weights,
            self.device,
        )
        self._board_extractor.eval()
        self._board_extractor.to(self.device)

    def _initialize_classifier(self) -> None:
        """Initialize the piece classifier model."""
        logger.info("Initializing piece classifier model...")
        self._classifier = ChessVision._get_classifier_model()
        self._classifier = ChessVision.load_model_checkpoint(self._classifier, self._classifier_weights, self.device)
        self._classifier.eval()
        self._classifier.to(self.device)

    def process_image(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
        flip: bool = False,
    ) -> ChessVisionResult:
        """Process a raw image and return complete results.

        Args:
            image: Input image as numpy array (BGR format)
            threshold: Threshold for board detection (0-1)
            flip: Whether to flip the board orientation

        Returns:
            ChessVisionResult containing all processing results
        """
        start_time = time.time()

        # Extract board
        board_result = self.extract_board(image, threshold)

        # Classify position if board was found
        position_result = None
        if board_result.board_image is not None:
            position_result = self.classify_position(board_result.board_image, flip)

        processing_time = time.time() - start_time

        return ChessVisionResult(
            board_extraction=board_result,
            position=position_result,
            processing_time=processing_time,
        )

    def extract_board(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
    ) -> BoardExtractionResult:
        """Extract chessboard from image.

        Args:
            image: Input image as numpy array (BGR format)
            threshold: Threshold for board detection (0-1)

        Returns:
            BoardExtractionResult containing extraction results
        """
        # Resize image
        comp_image = cv2.resize(image, ChessVision.INPUT_SIZE, interpolation=cv2.INTER_AREA)

        # Prepare image for model
        image_batch = torch.Tensor(np.array([comp_image])) / 255
        image_batch = image_batch.permute(0, 3, 1, 2).to(self.device)

        # Get model predictions
        with torch.no_grad():
            logits = self.board_extractor(image_batch)[0].squeeze().cpu().numpy()

        # Process predictions
        return self.process_board_extraction_logits(logits, image, threshold)

    def classify_position(
        self,
        board_image: np.ndarray,
        flip: bool = False,
    ) -> PositionResult:
        """Classify chess position from an extracted board image.

        Args:
            board_image: Extracted board image (grayscale)
            flip: Whether to flip the board orientation

        Returns:
            PositionResult containing classification results
        """
        # Extract individual squares
        squares, square_names = ChessVision._extract_squares(board_image, flip)

        # Prepare batch for model
        batch = torch.Tensor(squares).permute(0, 3, 1, 2).to(self.device)
        batch /= 255.0

        # Get predictions
        with torch.no_grad():
            predictions = self.classifier(batch)
            probabilities = F.softmax(predictions, dim=1)

        # Process results
        predictions = predictions.detach().cpu().numpy()
        probabilities = probabilities.detach().cpu().numpy()

        return self.process_classifier_logits(predictions, probabilities, square_names, squares)

    @staticmethod
    def process_board_extraction_logits(
        logits: np.ndarray,
        orig_image: np.ndarray,
        threshold: float,
    ) -> BoardExtractionResult:
        """Process board extraction logits without requiring model initialization.

        Args:
            logits: Raw model output logits
            orig_image: Original input image
            threshold: Threshold for board detection

        Returns:
            BoardExtractionResult containing extraction results
        """
        # Convert logits to probabilities
        probabilities = torch.sigmoid(torch.tensor(logits)).numpy()

        # Create binary mask
        binary_mask = ChessVision._create_binary_mask(probabilities, threshold)

        # Find quadrangle in mask
        quadrangle = ChessVision._find_quadrangle(binary_mask)

        # If no valid quadrangle found, return early
        if quadrangle is None:
            return BoardExtractionResult(
                board_image=None,
                logits=logits,
                probabilities=probabilities,
                binary_mask=binary_mask,
                quadrangle=None,
                confidence_scores={
                    "mask_mean": float(probabilities.mean()),
                    "mask_max": float(probabilities.max()),
                },
            )

        # Scale quadrangle to original image size
        scaled_quad = ChessVision._scale_quadrangle(quadrangle, (orig_image.shape[0], orig_image.shape[1]))

        # Extract and process board
        board = ChessVision._extract_perspective(orig_image, scaled_quad, ChessVision.BOARD_SIZE)
        if len(board.shape) == 3:  # If image has multiple channels
            board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        board = cv2.flip(board, 1)  # TODO: permute approximation instead

        return BoardExtractionResult(
            board_image=board,
            logits=logits,
            probabilities=probabilities,
            binary_mask=binary_mask,
            quadrangle=scaled_quad,
        )

    @staticmethod
    def process_classifier_logits(
        predictions: np.ndarray,
        probabilities: np.ndarray,
        square_names: list[str],
        squares: np.ndarray,
    ) -> PositionResult:
        """Process classifier logits without requiring model initialization.

        Args:
            predictions: Raw model output logits
            probabilities: Softmax probabilities
            square_names: Names of squares in order
            squares: Array of square images

        Returns:
            PositionResult containing classification results
        """
        # Calculate confidence scores
        confidence_scores = {name: float(prob.max()) for name, prob in zip(square_names, probabilities)}

        # Get initial predictions
        initial_predictions = np.argmax(predictions, axis=1)
        pred_labels = [ChessVision.LABEL_NAMES[p] for p in initial_predictions]

        # Apply chess logic to fix potential errors
        pred_labels = ChessVision._validate_position(pred_labels, predictions, square_names)

        # Create chess board from labels
        board = chess.BaseBoard(board_fen=None)
        for pred_label, sq in zip(pred_labels, square_names):
            piece = None if pred_label == "f" else chess.Piece.from_symbol(pred_label)
            square = chess.SQUARE_NAMES.index(sq)
            board.set_piece_at(square, piece, promoted=False)

        return PositionResult(
            fen=board.board_fen(promoted=False),
            predictions=predictions,
            squares=squares,
            square_names=square_names,
            confidence_scores=confidence_scores,
        )

    @staticmethod
    def _create_binary_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Convert probability mask to binary mask."""
        mask = mask.copy()  # Create a copy to avoid modifying the original
        mask[mask > threshold] = 255
        mask[mask <= threshold] = 0
        return mask.astype(np.uint8)

    @staticmethod
    def _find_quadrangle(mask: np.ndarray) -> np.ndarray | None:
        """Find a quadrangle (4-sided polygon) in a binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

        if len(contours) > 1:
            contours = ChessVision._filter_contours(mask.shape, contours)

        if not contours:
            return None

        # Try to approximate and find a quadrangle
        for contour in contours:
            arclen = cv2.arcLength(contour, True)
            candidate = cv2.approxPolyDP(contour, 0.1 * arclen, True)

            if len(candidate) == 4:
                return ChessVision._rotate_quadrangle(candidate)

        return None

    @staticmethod
    def _filter_contours(
        img_shape: tuple[int, int],
        contours: list[np.ndarray],
        min_ratio_bounding: float = 0.6,
        min_area_percentage: float = 0.35,
        max_area_percentage: float = 1.0,
    ) -> list[np.ndarray]:
        """Filter contours based on area and aspect ratio criteria."""
        filtered = []
        mask_area = float(img_shape[0] * img_shape[1])

        for contour in contours:
            area = cv2.contourArea(contour) / mask_area
            if area < min_area_percentage or area > max_area_percentage:
                continue

            _, _, w, h = cv2.boundingRect(contour)
            if ChessVision._ratio(h, w) < min_ratio_bounding:
                continue

            filtered.append(contour)

        return filtered

    @staticmethod
    def _rotate_quadrangle(approx: np.ndarray) -> np.ndarray:
        """Rotate quadrangle to ensure consistent orientation."""
        if approx[0, 0, 0] < approx[2, 0, 0]:
            approx = approx[[3, 0, 1, 2], :, :]
        return approx

    @staticmethod
    def _scale_quadrangle(approx: np.ndarray, orig_size: tuple[int, int]) -> np.ndarray:
        """Scale quadrangle approximation to match original image size."""
        sf = orig_size[0] / 256.0
        return np.array(approx * sf, dtype=np.uint32)

    @staticmethod
    def _extract_perspective(
        image: np.ndarray,
        approx: np.ndarray,
        out_size: tuple[int, int],
    ) -> np.ndarray:
        """Extract a perspective-corrected region from an image."""
        w, h = out_size[0], out_size[1]
        dest = np.array(((0, 0), (w, 0), (w, h), (0, h)), np.float32)
        approx = np.array(approx, np.float32)

        coeffs = cv2.getPerspectiveTransform(approx, dest)
        return cv2.warpPerspective(image, coeffs, out_size)

    @staticmethod
    def _extract_squares(
        board: np.ndarray,
        flip: bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        """Extract individual squares from board image.

        Args:
            board: A 512x512 image of a chessboard
            flip: Whether to flip the board orientation

        Returns:
            Tuple containing:
                - Array of square images (64, 64, 64, 1)
                - List of square names (e.g. ['a8', 'b8', ...])
        """
        ranks = ["a", "b", "c", "d", "e", "f", "g", "h"]
        files = ["1", "2", "3", "4", "5", "6", "7", "8"]

        if flip:
            ranks = list(reversed(ranks))
            files = list(reversed(files))

        squares_list = []
        names = []

        # Calculate square size
        ww, hh = board.shape
        w = int(ww / 8)
        h = int(hh / 8)

        # Extract each square
        for i in range(8):
            for j in range(8):
                square = board[i * w : (i + 1) * w, j * h : (j + 1) * h]
                squares_list.append(square)
                names.append(ranks[j] + files[7 - i])

        # Reshape squares to (N, 64, 64, 1)
        squares = np.array(squares_list)
        squares = squares.reshape(squares.shape[0], 64, 64, 1)

        return squares, names

    @staticmethod
    def _validate_position(
        pred_labels: list[str],
        probabilities: np.ndarray,
        square_names: list[str],
    ) -> list[str]:
        """Apply chess logic to validate and fix predictions.

        Args:
            pred_labels: Initial piece predictions for each square
            probabilities: Raw model probabilities
            square_names: Names of squares in order

        Returns:
            Validated and potentially corrected piece labels
        """
        # Get sorted probabilities for each square
        argsorted_probs = np.argsort(probabilities)
        sorted_probs = np.take_along_axis(probabilities, argsorted_probs, axis=-1)

        # Fix pawns on first/last rank
        for i, (label, name) in enumerate(zip(pred_labels, square_names)):
            if name in ChessVision.INVALID_PAWN_SQUARES and label in ["P", "p"]:
                # Get next best prediction that isn't a pawn
                for alt_idx in argsorted_probs[i][::-1]:
                    alt_piece = ChessVision.LABEL_NAMES[alt_idx]
                    if alt_piece not in ["P", "p"]:
                        pred_labels[i] = alt_piece
                        break

        # Fix bishops (no more than one per color square per side)
        white_bishops: dict[str, list[tuple[int, float]]] = {"dark": [], "light": []}
        black_bishops: dict[str, list[tuple[int, float]]] = {"dark": [], "light": []}

        # Find all bishops
        for i, (label, name) in enumerate(zip(pred_labels, square_names)):
            if label == "B":
                color = "dark" if name in ChessVision.DARK_SQUARES else "light"
                white_bishops[color].append((i, sorted_probs[i][-1]))
            elif label == "b":
                color = "dark" if name in ChessVision.DARK_SQUARES else "light"
                black_bishops[color].append((i, sorted_probs[i][-1]))

        # Fix duplicate bishops
        for bishops in [white_bishops, black_bishops]:
            for color in ["dark", "light"]:
                if len(bishops[color]) > 1:
                    # Keep the bishop with highest confidence
                    bishops[color].sort(key=lambda x: x[1], reverse=True)
                    # Change others to next best prediction
                    for idx, _ in bishops[color][1:]:
                        for alt_idx in argsorted_probs[idx][::-1]:
                            alt_piece = ChessVision.LABEL_NAMES[alt_idx]
                            if alt_piece not in ["B", "b"]:
                                pred_labels[idx] = alt_piece
                                break

        return pred_labels

    @staticmethod
    def _get_classifier_model() -> torch.nn.Module:
        """Initialize the piece classifier model.

        Returns:
            ResNet18 model configured for chess piece classification
        """
        return timm.create_model(ChessVision.MODEL_ID, num_classes=ChessVision.NUM_CLASSES, in_chans=1)

    @classmethod
    def load_model_checkpoint(
        cls,
        model: torch.nn.Module,
        checkpoint_path: str,
        device: torch.device | None = None,
    ) -> torch.nn.Module:
        """Load a model checkpoint.

        This is a convenience method for loading model checkpoints during training
        or evaluation. It handles both old and new checkpoint formats.

        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint file
            device: Optional device to load the checkpoint to

        Returns:
            Model with loaded weights
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                # New format with metadata
                model.load_state_dict(state_dict["model_state_dict"])
                metadata = state_dict.get("metadata", {})
            elif "state_dict" in state_dict:
                # timm format
                model.load_state_dict(state_dict["state_dict"])
                metadata = state_dict.get("metadata", {})
            else:
                # Old format - state_dict is directly the model weights
                # or has "model" key instead of "model_state_dict"
                if "model" in state_dict:
                    model_weights = state_dict["model"]
                    metadata = state_dict.get("metadata", {})
                else:
                    model_weights = state_dict
                    metadata = {}
                model.load_state_dict(model_weights)
        else:
            # Direct state dict
            model.load_state_dict(state_dict)
            metadata = {}

        if metadata:
            model.metadata = metadata

        return model

"""Core ChessVision functionality."""

from __future__ import annotations

import logging
import time

import chess
import cv2
import numpy as np
import timm
import torch
from numpy.typing import NDArray
from torch.nn.functional import softmax

from . import constants, utils
from .cv_types import BoardExtractionResult, ChessVisionResult, PositionResult
from .pytorch_unet.unet.unet_model import UNet

logger = logging.getLogger(__name__)


class ChessVision:
    """Main class for chess position detection from images."""

    def __init__(
        self,
        board_extractor_weights: str | None = None,
        classifier_weights: str | None = None,
        classifier_model_id: str | None = None,
        lazy_load: bool = True,
    ):
        """Initialize ChessVision with optional custom model weights.

        Args:
            board_extractor_weights: Path to board extraction model weights.
                                   If None, uses best available weights.
            classifier_weights: Path to piece classifier model weights.
                              If None, uses best available weights.
            classifier_model_id: Model architecture to use. If None, tries YOLO first,
                               falling back to ResNet if YOLO is not available.
                               If "yolo" is specified, fails if YOLO is not available.
                               Other values are passed directly to timm.
            lazy_load: If True, models are loaded only when needed.
                      If False, models are loaded immediately.
        """
        logger.info("Initializing ChessVision instance...")
        self.device = utils.get_device()
        self._board_extractor: torch.nn.Module | None = None
        self._classifier: torch.nn.Module | None = None
        self._board_extractor_weights = board_extractor_weights or constants.BEST_EXTRACTOR_WEIGHTS

        self._classifier_weights = classifier_weights
        self._classifier_model_id = classifier_model_id

        if not lazy_load:
            logger.info("Eager loading models...")
            self._initialize_board_extractor()
            self._initialize_classifier()
            logger.info("Models loaded successfully")

    @property
    def board_extractor(self) -> torch.nn.Module:
        """Get the board extractor model, initializing if needed."""
        if self._board_extractor is None:
            self._initialize_board_extractor()
        assert self._board_extractor is not None
        return self._board_extractor

    @property
    def classifier(self) -> torch.nn.Module:
        """Get the classifier model, initializing if needed."""
        if self._classifier is None:
            self._initialize_classifier()
        assert self._classifier is not None
        return self._classifier

    def _initialize_board_extractor(self) -> None:
        """Initialize the board extraction model."""
        logger.info("Initializing board extraction model...")
        self._board_extractor = UNet(n_channels=3, n_classes=1)
        self._board_extractor = self._board_extractor.to(memory_format=torch.channels_last)  # type: ignore
        self._board_extractor = utils.load_model_checkpoint(
            self._board_extractor,  # type: ignore
            self._board_extractor_weights,
            self.device,
        )
        self._board_extractor.eval()
        self._board_extractor.to(self.device)

    def _initialize_classifier(self) -> None:
        """Initialize the piece classifier model."""
        logger.info("Initializing piece classifier model...")

        # If no model specified, try YOLO first
        if self._classifier_model_id is None:
            try:
                self._classifier = utils.load_yolo_model(self._classifier_weights or constants.BEST_YOLO_CLASSIFIER)
                self._classifier_model_id = "yolo"  # Mark as using YOLO
                self._classifier_weights = self._classifier_weights or constants.BEST_YOLO_CLASSIFIER
                logger.info(f"Loaded YOLO model from {self._classifier_weights or constants.BEST_YOLO_CLASSIFIER}")
            except ImportError:
                logger.info("YOLO not available, falling back to ResNet18")
                self._classifier = utils.get_classifier_model(self._classifier_model_id or "resnet18")
                self._classifier = utils.load_model_checkpoint(
                    self._classifier,
                    self._classifier_weights or constants.BEST_CLASSIFIER_WEIGHTS,
                    self.device,
                )
                self._classifier_model_id = "resnet18"
                self._classifier_weights = self._classifier_weights or constants.BEST_CLASSIFIER_WEIGHTS
        # If YOLO explicitly requested, try loading it or fail
        elif self._classifier_model_id == "yolo":
            self._classifier = utils.load_yolo_model(self._classifier_weights or constants.BEST_YOLO_CLASSIFIER)
            self._classifier_weights = self._classifier_weights or constants.BEST_YOLO_CLASSIFIER
            logger.info(f"Loaded YOLO model from {self._classifier_weights}")
        # Otherwise load the specified model through timm
        else:
            self._classifier = utils.get_classifier_model(self._classifier_model_id)
            self._classifier = utils.load_model_checkpoint(
                self._classifier,
                self._classifier_weights or constants.BEST_CLASSIFIER_WEIGHTS,
                self.device,
            )
            self._classifier_weights = self._classifier_weights or constants.BEST_CLASSIFIER_WEIGHTS

        assert self._classifier is not None
        self._classifier.eval()
        self._classifier.to(self.device)

    def process_image(
        self,
        image: NDArray[np.uint8],
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
        assert isinstance(image, np.ndarray), "Image must be a numpy array"
        assert image.dtype == np.uint8, "Image must be uint8"
        assert len(image.shape) == 3, "Image must be 3-dimensional (H,W,C)"

        logger.info("Starting image processing pipeline...")
        start_time = time.time()

        # Extract board
        board_result = self.extract_board(image, threshold)
        if board_result.board_image is None:
            logger.info("No valid board found in image")
        else:
            logger.info("Board successfully extracted")

        # Classify position if board was found
        position_result = None
        if board_result.board_image is not None:
            position_result = self.classify_position(board_result.board_image, flip)
            logger.info("Position classification completed")

        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")

        return ChessVisionResult(
            board_extraction=board_result,
            position=position_result,
            processing_time=processing_time,
        )

    def extract_board(
        self,
        image: NDArray[np.uint8],
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
        comp_image = cv2.resize(image, constants.INPUT_SIZE, interpolation=cv2.INTER_AREA)

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
        board_image: NDArray[np.uint8],
        flip: bool = False,
    ) -> PositionResult:
        """Classify chess position from an extracted board image.

        Args:
            board_image: Extracted board image (grayscale)
            flip: Whether to flip the board orientation

        Returns:
            PositionResult containing classification results
        """
        # Extract individual squares and get square names
        squares = self.extract_squares(board_image, flip)
        square_names = self._get_square_names(flip)

        # Prepare batch for model
        batch = torch.Tensor(squares).permute(0, 3, 1, 2).to(self.device)
        batch /= 255.0

        # Get predictions
        with torch.no_grad():
            predictions = self.classifier(batch)
            probabilities = predictions if self._classifier_model_id == "yolo" else softmax(predictions, dim=1)

        # Process results
        predictions_np = predictions.detach().cpu().numpy()
        probabilities_np = probabilities.detach().cpu().numpy()

        return self.process_classifier_logits(
            predictions_np,
            probabilities_np,
            square_names,
            squares,
        )

    @staticmethod
    def process_board_extraction_logits(
        logits: NDArray[np.float32],
        orig_image: NDArray[np.uint8],
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
        assert isinstance(logits, np.ndarray), "Logits must be a numpy array"
        assert logits.dtype == np.float32, "Logits must be float32"
        assert isinstance(orig_image, np.ndarray), "Original image must be a numpy array"
        assert orig_image.dtype == np.uint8, "Original image must be uint8"

        # Convert logits to probabilities
        probabilities = torch.sigmoid(torch.tensor(logits)).numpy()

        # Create binary mask
        binary_mask = utils.create_binary_mask(probabilities, threshold)

        # Find quadrangle in mask
        quadrangle = ChessVision._find_quadrangle(binary_mask)

        if quadrangle is None:
            logger.info("Failed to extract board from image")
            return BoardExtractionResult(
                board_image=None,
                logits=logits,
                probabilities=probabilities,
                binary_mask=binary_mask,
                quadrangle=None,
            )

        # Scale quadrangle to original image size
        scaled_quad = ChessVision._scale_quadrangle(
            quadrangle,
            (orig_image.shape[0], orig_image.shape[1]),
        )
        assert scaled_quad.dtype == np.float32, "Scaled quadrangle must be float32"

        # Extract and process board
        board = utils.extract_perspective(orig_image, scaled_quad, constants.BOARD_SIZE)
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
        predictions: NDArray[np.float32],
        probabilities: NDArray[np.float32],
        square_names: list[str],
        squares: NDArray[np.uint8],
    ) -> PositionResult:
        """Process classifier predictions and probabilities.

        Args:
            predictions: Model predictions (logits for ResNet, probabilities for YOLO)
            probabilities: Softmax probabilities
            square_names: Names of squares in order
            squares: Array of square images

        Returns:
            PositionResult containing classification results
        """
        # Calculate confidence scores using probabilities
        confidence_scores = {name: float(prob.max()) for name, prob in zip(square_names, probabilities)}

        # Get initial predictions from probabilities
        initial_predictions = np.argmax(probabilities, axis=1)
        pred_labels = [constants.LABEL_NAMES[p] for p in initial_predictions]

        # Apply chess logic to fix potential errors
        pred_labels = ChessVision.validate_position(pred_labels, probabilities, square_names)

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
    def _find_quadrangle(mask: NDArray[np.uint8]) -> NDArray[np.int32] | None:
        """Find a quadrangle (4-sided polygon) in a binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

        if len(contours) > 1:
            contours = ChessVision._filter_contours(
                (mask.shape[0], mask.shape[1]),
                contours,  # type: ignore
            )

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
        contours: list[NDArray[np.int32]],
        min_ratio_bounding: float = 0.6,
        min_area_percentage: float = 0.35,
        max_area_percentage: float = 1.0,
    ) -> list[NDArray[np.int32]]:
        """Filter contours based on area and aspect ratio criteria."""
        filtered = []
        mask_area = float(img_shape[0] * img_shape[1])

        for contour in contours:
            area = cv2.contourArea(contour) / mask_area
            if area < min_area_percentage or area > max_area_percentage:
                continue

            _, _, w, h = cv2.boundingRect(contour)
            if utils.ratio(h, w) < min_ratio_bounding:
                continue

            filtered.append(contour)

        return filtered

    @staticmethod
    def _rotate_quadrangle(approx: NDArray[np.int32]) -> NDArray[np.int32]:
        """Rotate quadrangle to ensure consistent orientation."""
        if approx[0, 0, 0] < approx[2, 0, 0]:
            approx = approx[[3, 0, 1, 2], :, :]
        return approx

    @staticmethod
    def _scale_quadrangle(approx: NDArray[np.int32], orig_size: tuple[int, int]) -> NDArray[np.float32]:
        """Scale quadrangle approximation to match original image size."""
        sf = orig_size[0] / 256.0
        return np.array(approx * sf, dtype=np.float32)

    @staticmethod
    def _get_square_names(flip: bool = False) -> list[str]:
        """Get the list of square names in standard chess notation.

        Args:
            flip: Whether to flip the board orientation

        Returns:
            List of square names (e.g. ['a8', 'b8', ...])
        """
        ranks = ["a", "b", "c", "d", "e", "f", "g", "h"]
        files = ["1", "2", "3", "4", "5", "6", "7", "8"]

        if flip:
            ranks = list(reversed(ranks))
            files = list(reversed(files))

        names = []
        for i in range(8):
            for j in range(8):
                names.append(ranks[j] + files[7 - i])

        return names

    @staticmethod
    def extract_squares(
        board: NDArray[np.uint8],
        flip: bool = False,
    ) -> NDArray[np.uint8]:
        """Extract individual squares from board image.

        Args:
            board: A 512x512 image of a chessboard
            flip: Whether to flip the board orientation

        Returns:
            Array of square images (64, 64, 64, 1)
        """
        # Calculate square size
        h, w = board.shape
        square_h, square_w = h // 8, w // 8

        # Create a view of the board as 8x8 grid of squares
        squares = board.reshape(8, square_h, 8, square_w)
        squares = squares.transpose(0, 2, 1, 3)  # Reorder to get squares in correct order
        squares = squares.reshape(64, square_h, square_w)  # Flatten to 64 squares

        if flip:
            # Reverse both rows and columns order when flipping
            squares = squares.reshape(8, 8, square_h, square_w)
            squares = squares[::-1, ::-1]  # Flip both dimensions
            squares = squares.reshape(64, square_h, square_w)

        # Add channel dimension
        squares = squares.reshape(64, square_h, square_w, 1)

        return squares

    @staticmethod
    def validate_position(
        pred_labels: list[str],
        probabilities: NDArray[np.float32],
        square_names: list[str],s,
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
            if name in constants.INVALID_PAWN_SQUARES and label in ["P", "p"]:
                # Get next best prediction that isn't a pawn
                for alt_idx in argsorted_probs[i][::-1]:
                    alt_piece = constants.LABEL_NAMES[alt_idx]
                    if alt_piece not in ["P", "p"]:
                        pred_labels[i] = alt_piece
                        break

        # Fix bishops (no more than one per color square per side)
        white_bishops: dict[str, list[tuple[int, float]]] = {"dark": [], "light": []}
        black_bishops: dict[str, list[tuple[int, float]]] = {"dark": [], "light": []}

        # Find all bishops
        for i, (label, name) in enumerate(zip(pred_labels, square_names)):
            if label == "B":
                color = "dark" if name in constants.DARK_SQUARES else "light"
                white_bishops[color].append((i, sorted_probs[i][-1]))
            elif label == "b":
                color = "dark" if name in constants.DARK_SQUARES else "light"
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
                            alt_piece = constants.LABEL_NAMES[alt_idx]
                            if alt_piece not in ["B", "b"]:
                                pred_labels[idx] = alt_piece
                                break

        return pred_labels

    @staticmethod
    def get_classifier_model(model_id: str = "resnet18") -> torch.nn.Module:
        """Initialize the piece classifier model.

        Returns:
            ResNet18 model configured for chess piece classification
        """
        return timm.create_model(  # type: ignore
            model_id,
            num_classes=constants.NUM_CLASSES,
            in_chans=1,
        )

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
            device = utils.get_device()

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

"""Core ChessVision functionality."""

from __future__ import annotations

import logging
import time

import chess
import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn.functional import softmax

from . import constants, utils
from .cv_types import BoardExtractionResult, ChessVisionResult, PositionResult, ValidationFix
from .pytorch_unet.unet.unet_model import UNet

logger = logging.getLogger(__name__)


class ChessVision:
    """Main class for chess position detection from images."""

    def __init__(
        self,
        board_extractor_weights: str | None = None,
        board_extractor_model_id: str | None = None,
        classifier_weights: str | None = None,
        classifier_model_id: str | None = None,
        lazy_load: bool = True,
    ):
        """Initialize ChessVision with optional custom model weights.

        Args:
            board_extractor_weights: Path to board extraction model weights.
                                   If None, uses best available weights.
            board_extractor_model_id: Model architecture to use. If None, tries YOLO first,
                                     falling back to UNet if YOLO is not available.
                                     If "yolo" is specified, fails if YOLO is not available.
                                     Other values are passed directly to timm.
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
        self._board_extractor_weights = board_extractor_weights
        self._board_extractor_model_id = board_extractor_model_id
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
        if hasattr(self._classifier, "metadata"):
            logger.info(f"Classifier metadata: {self._classifier.metadata}")
        return self._classifier

    def _initialize_board_extractor(self) -> None:
        """Initialize the board extraction model."""
        logger.info("Initializing board extraction model...")
        if self._board_extractor_model_id is None:
            self._board_extractor = UNet(n_channels=3, n_classes=1)
            use_channels_last = self.device.type in ["cuda", "cpu"]
            if use_channels_last:
                self._board_extractor = self._board_extractor.to(memory_format=torch.channels_last)  # type: ignore
            self._board_extractor = utils.load_model_checkpoint(
                self._board_extractor,  # type: ignore
                self._board_extractor_weights,
                self.device,
            )
        elif self._board_extractor_model_id == "yolo":
            self._board_extractor = utils.load_yolo_segmentation_model(
                self._board_extractor_weights or constants.BEST_YOLO_EXTRACTOR,
            )
        else:
            assert False, f"Invalid board extractor model ID: {self._board_extractor_model_id}"

        if hasattr(self._board_extractor, "metadata"):
            logger.info(f"Board extractor metadata: {self._board_extractor.metadata}")

        self._board_extractor.eval()
        self._board_extractor.to(self.device)

    def _initialize_classifier(self) -> None:
        """Initialize the piece classifier model."""
        logger.info("Initializing piece classifier model...")

        # If no model specified, try YOLO first
        if self._classifier_model_id is None:
            try:
                self._classifier = utils.load_yolo_classification_model(
                    self._classifier_weights or constants.BEST_YOLO_CLASSIFIER
                )
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
            self._classifier = utils.load_yolo_classification_model(
                self._classifier_weights or constants.BEST_YOLO_CLASSIFIER
            )
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
        """Classify chess position from an extracted board image."""
        # Extract individual squares and get square names
        squares = self.extract_squares(board_image)
        square_names = constants.SQUARE_NAMES_FLIPPED if flip else constants.SQUARE_NAMES_NORMAL

        # Prepare batch for model
        batch = torch.Tensor(squares).permute(0, 3, 1, 2).to(self.device)
        batch /= 255.0

        # Get predictions
        with torch.no_grad():
            predictions = self.classifier(batch)
            probabilities = predictions if self._classifier_model_id == "yolo" else softmax(predictions, dim=1)
            probabilities_np = probabilities.detach().cpu().numpy()

        return self.process_position_probabilities(
            probabilities=probabilities_np,
            square_names=square_names,
            square_crops=squares,
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
                binary_mask=binary_mask,
                quadrangle=None,
                probabilities=logits,
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
            binary_mask=binary_mask,
            quadrangle=scaled_quad,
            probabilities=logits,
        )

    @staticmethod
    def process_position_probabilities(
        probabilities: NDArray[np.float32],
        square_names: list[str],
        square_crops: NDArray[np.uint8],
    ) -> PositionResult:
        """Process model probabilities into a chess position with validation.

        Args:
            probabilities: Model probability distributions (64, 13)
            square_names: Names of squares in order (a8-h8, a7-h7, ..., a1-h1)
            square_crops: Array of square images (64, 64, 64, 1)

        Returns:
            PositionResult containing both original and validated positions
        """
        # Get initial predictions from probabilities
        initial_predictions = np.argmax(probabilities, axis=1)
        pred_labels = [constants.LABEL_NAMES[p] for p in initial_predictions]

        # Create initial board
        board = chess.BaseBoard(board_fen=None)
        for pred_label, sq in zip(pred_labels, square_names):
            piece = None if pred_label == "f" else chess.Piece.from_symbol(pred_label)
            square = chess.SQUARE_NAMES.index(sq)
            board.set_piece_at(square, piece, promoted=False)

        original_fen = board.board_fen(promoted=False)

        # Apply validation rules
        validated_labels, fixes = ChessVision.validate_position(pred_labels, probabilities, square_names)

        # Create final board with validated position
        board = chess.BaseBoard(board_fen=None)
        for pred_label, sq in zip(validated_labels, square_names):
            piece = None if pred_label == "f" else chess.Piece.from_symbol(pred_label)
            square = chess.SQUARE_NAMES.index(sq)
            board.set_piece_at(square, piece, promoted=False)

        return PositionResult(
            fen=board.board_fen(promoted=False),
            original_fen=original_fen,
            model_probabilities=probabilities,  # Store raw model probabilities
            squares=square_crops,
            square_names=square_names,
            validation_fixes=fixes,
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
    def extract_squares(
        board: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Extract individual squares from board image.

        Args:
            board: A 512x512 grayscale image of a chessboard

        Returns:
            Array of square images (64, 64, 64, 1)
        """
        # Calculate square size
        h, w = board.shape
        square_h, square_w = h // 8, w // 8

        # Create a view of the board as 8x8 grid of squares
        squares = board.reshape(8, square_h, 8, square_w)
        squares = squares.transpose(0, 2, 1, 3)  # Reorder to get squares in correct order
        squares = squares.reshape(64, square_h, square_w)
        return squares.reshape(64, square_h, square_w, 1)

    @staticmethod
    def validate_position(
        pred_labels: list[str],
        probabilities: NDArray[np.float32],
        square_names: list[str],
    ) -> tuple[list[str], list[ValidationFix]]:
        """Apply chess rules to validate and fix predictions."""
        fixes: list[ValidationFix] = []

        # Get sorted probabilities for each square
        argsorted_probs = np.argsort(probabilities)

        # Rule 1: No pawns on first/last rank
        for i, (label, name) in enumerate(zip(pred_labels, square_names)):
            if name in constants.INVALID_PAWN_SQUARES and label in ["P", "p"]:
                # Find next best non-pawn prediction
                for alt_idx in argsorted_probs[i][::-1]:
                    alt_piece = constants.LABEL_NAMES[alt_idx]
                    if alt_piece not in ["P", "p"]:
                        fixes.append(
                            ValidationFix(
                                square_name=name,
                                original_piece=label,
                                corrected_piece=alt_piece,
                                rule_name="no_pawns_on_ends",
                            ),
                        )
                        pred_labels[i] = alt_piece
                        break

        # # Rule 2: One king per color
        # # Find all white kings
        # white_king_squares = [
        #     (i, probabilities[i, constants.LABEL_INDICES["K"]]) for i, label in enumerate(pred_labels) if label == "K"
        # ]
        # if len(white_king_squares) > 1:
        #     # Sort by probability, keep the most likely king
        #     white_king_squares.sort(key=lambda x: x[1], reverse=True)
        #     # Fix all but the most likely king
        #     for square_idx, _ in white_king_squares[1:]:
        #         # Find next best non-king prediction
        #         for alt_idx in argsorted_probs[square_idx][::-1]:
        #             alt_piece = constants.LABEL_NAMES[alt_idx]
        #             if alt_piece != "K":
        #                 fixes.append(
        #                     ValidationFix(
        #                         square_name=square_names[square_idx],
        #                         original_piece="K",
        #                         corrected_piece=alt_piece,
        #                         rule_name="one_king_per_color",
        #                     ),
        #                 )
        #                 pred_labels[square_idx] = alt_piece
        #                 break

        # # Same for black kings
        # black_king_squares = [
        #     (i, probabilities[i, constants.LABEL_INDICES["k"]]) for i, label in enumerate(pred_labels) if label == "k"
        # ]
        # if len(black_king_squares) > 1:
        #     black_king_squares.sort(key=lambda x: x[1], reverse=True)
        #     for square_idx, _ in black_king_squares[1:]:
        #         for alt_idx in argsorted_probs[square_idx][::-1]:
        #             alt_piece = constants.LABEL_NAMES[alt_idx]
        #             if alt_piece != "k":
        #                 fixes.append(
        #                     ValidationFix(
        #                         square_name=square_names[square_idx],
        #                         original_piece="k",
        #                         corrected_piece=alt_piece,
        #                         rule_name="one_king_per_color",
        #                     ),
        #                 )
        #                 pred_labels[square_idx] = alt_piece
        #                 break

        # # Rule 3: Maximum two bishops per color
        # # Handle white bishops
        # white_bishop_squares = [
        #     (i, probabilities[i, constants.LABEL_INDICES["B"]]) for i, label in enumerate(pred_labels) if label == "B"
        # ]
        # if len(white_bishop_squares) > 2:
        #     # Sort by probability, keep the two most likely bishops
        #     white_bishop_squares.sort(key=lambda x: x[1], reverse=True)
        #     # Fix all extra bishops beyond the first two
        #     for square_idx, _ in white_bishop_squares[2:]:
        #         # Find next best non-bishop prediction
        #         for alt_idx in argsorted_probs[square_idx][::-1]:
        #             alt_piece = constants.LABEL_NAMES[alt_idx]
        #             if alt_piece != "B":
        #                 fixes.append(
        #                     ValidationFix(
        #                         square_name=square_names[square_idx],
        #                         original_piece="B",
        #                         corrected_piece=alt_piece,
        #                         rule_name="max_two_bishops",
        #                     ),
        #                 )
        #                 pred_labels[square_idx] = alt_piece
        #                 break

        # # Handle black bishops
        # black_bishop_squares = [
        #     (i, probabilities[i, constants.LABEL_INDICES["b"]]) for i, label in enumerate(pred_labels) if label == "b"
        # ]
        # if len(black_bishop_squares) > 2:
        #     black_bishop_squares.sort(key=lambda x: x[1], reverse=True)
        #     for square_idx, _ in black_bishop_squares[2:]:
        #         for alt_idx in argsorted_probs[square_idx][::-1]:
        #             alt_piece = constants.LABEL_NAMES[alt_idx]
        #             if alt_piece != "b":
        #                 fixes.append(
        #                     ValidationFix(
        #                         square_name=square_names[square_idx],
        #                         original_piece="b",
        #                         corrected_piece=alt_piece,
        #                         rule_name="max_two_bishops",
        #                     ),
        #                 )
        #                 pred_labels[square_idx] = alt_piece
        #                 break

        # Debug print fixes:
        # for fix in fixes:
        #     print(f"{fix.square_name} {fix.original_piece} -> {fix.corrected_piece}")

        return pred_labels, fixes

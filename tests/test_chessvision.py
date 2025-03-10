"""High-level API tests for ChessVision."""

import cv2
import numpy as np
import pytest

from chessvision import ChessVision, constants


@pytest.fixture
def cv_model() -> ChessVision:
    """Create a ChessVision instance with default weights."""
    return ChessVision()


@pytest.fixture
def test_image() -> np.ndarray:
    """Load a test image from the test data directory."""
    test_image_path = constants.DATA_ROOT / "test" / "raw" / "1bf29f73-bc30-448b-a894-bd6428754a0c.JPG"
    if not test_image_path.exists():
        pytest.skip(f"Test image not found at {test_image_path}")
    return cv2.imread(str(test_image_path))


def test_chessvision_initialization() -> None:
    """Test that ChessVision can be initialized with default and custom weights."""
    # Test default initialization
    cv = ChessVision()
    assert cv._board_extractor is None  # Should be None due to lazy loading
    assert cv._classifier is None  # Should be None due to lazy loading
    assert cv._board_extractor_weights == constants.BEST_EXTRACTOR_WEIGHTS
    assert cv._classifier_weights == constants.BEST_CLASSIFIER_WEIGHTS

    # Test custom weights initialization
    custom_extractor = "path/to/extractor.pth"
    custom_classifier = "path/to/classifier.pth"
    cv = ChessVision(
        board_extractor_weights=custom_extractor,
        classifier_weights=custom_classifier,
    )
    assert cv._board_extractor_weights == custom_extractor
    assert cv._classifier_weights == custom_classifier


def test_process_image(cv_model: ChessVision, test_image: np.ndarray) -> None:
    """Test the main image processing pipeline."""
    result = cv_model.process_image(test_image)

    # Check that we got a result
    assert result is not None

    # Check board extraction result
    assert result.board_extraction is not None
    assert isinstance(result.board_extraction.binary_mask, np.ndarray)
    assert result.board_extraction.binary_mask.dtype == np.uint8

    # If board was found, check position result
    if result.board_extraction.board_image is not None:
        assert result.position is not None
        assert result.position.fen is not None
        assert len(result.position.predictions) == 64  # One prediction per square
        assert len(result.position.squares) == 64  # One image per square
        assert len(result.position.square_names) == 64  # One name per square
        assert all(name in result.position.confidence_scores for name in result.position.square_names)

    # Check processing time
    assert result.processing_time > 0


def test_extract_board(cv_model: ChessVision, test_image: np.ndarray) -> None:
    """Test board extraction specifically."""
    result = cv_model.extract_board(test_image)

    assert result is not None
    assert isinstance(result.binary_mask, np.ndarray)
    assert result.binary_mask.dtype == np.uint8
    assert result.logits is not None
    assert result.probabilities is not None

    if result.board_image is not None:
        assert isinstance(result.board_image, np.ndarray)
        assert result.board_image.shape == (512, 512)  # Should match BOARD_SIZE
        assert result.quadrangle is not None


def test_classify_position(cv_model: ChessVision, test_image: np.ndarray) -> None:
    """Test position classification with a known board image."""
    # First extract the board
    board_result = cv_model.extract_board(test_image)
    if board_result.board_image is None:
        pytest.skip("Could not extract board from test image")

    # Then classify the position
    result = cv_model.classify_position(board_result.board_image)

    assert result is not None
    assert result.fen is not None
    assert len(result.predictions) == 64
    assert len(result.squares) == 64
    assert len(result.square_names) == 64
    assert all(name in result.confidence_scores for name in result.square_names)

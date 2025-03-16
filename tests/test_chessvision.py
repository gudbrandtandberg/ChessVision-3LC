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
    test_image_path = constants.DATA_ROOT / "test" / "initial" / "raw" / "1bf29f73-bc30-448b-a894-bd6428754a0c.JPG"
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
    assert cv._classifier_weights is None

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
        assert result.position.original_fen is not None  # Check original FEN
        assert result.position.model_probabilities is not None  # Updated name
        assert result.position.squares is not None
        assert result.position.square_names is not None
        assert result.position.validation_fixes is not None  # Check validation fixes

    # Check processing time
    assert result.processing_time > 0


def test_extract_board(cv_model: ChessVision, test_image: np.ndarray) -> None:
    """Test board extraction specifically."""
    result = cv_model.extract_board(test_image)

    assert result is not None
    assert isinstance(result.binary_mask, np.ndarray)
    assert result.binary_mask.dtype == np.uint8
    assert result.quadrangle is not None

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
    assert result.original_fen is not None  # Check original FEN
    assert result.model_probabilities is not None  # Updated name
    assert result.squares is not None
    assert result.square_names is not None
    assert result.validation_fixes is not None  # Check validation fixes

    # Check that validation fixes are properly structured
    if len(result.validation_fixes) > 0:
        fix = result.validation_fixes[0]
        assert fix.square_name in result.square_names
        assert fix.original_piece in constants.LABEL_NAMES
        assert fix.corrected_piece in constants.LABEL_NAMES
        assert isinstance(fix.rule_name, str)

    # Check that original and validated FENs differ if there are fixes
    if result.validation_fixes:
        assert result.original_fen != result.fen
    else:
        assert result.original_fen == result.fen


def test_extract_squares() -> None:
    """Test that squares are correctly extracted from a board image."""
    # Create a test board image (512x512) with unique values for each square
    board = np.zeros((512, 512), dtype=np.uint8)
    square_size = 64

    # Fill each square with a unique value based on its position
    for rank in range(8):
        for file in range(8):
            value = rank * 8 + file
            board[rank * square_size : (rank + 1) * square_size, file * square_size : (file + 1) * square_size] = value

    # Test extraction (default orientation - white's perspective)
    squares = ChessVision.extract_squares(board)
    assert squares.shape == (64, 64, 64, 1)

    # Verify square positions in standard orientation
    # a8 is index 0, h8 is index 7, a1 is index 56, h1 is index 63
    assert squares[0, 0, 0, 0] == 0  # a8 should be value 0 (top-left)
    assert squares[7, 0, 0, 0] == 7  # h8 should be value 7 (top-right)
    assert squares[56, 0, 0, 0] == 56  # a1 should be value 56 (bottom-left)
    assert squares[63, 0, 0, 0] == 63  # h1 should be value 63 (bottom-right)

    # Test middle squares
    assert squares[8, 0, 0, 0] == 8  # a7
    assert squares[15, 0, 0, 0] == 15  # h7
    assert squares[16, 0, 0, 0] == 16  # a6
    assert squares[23, 0, 0, 0] == 23  # h6

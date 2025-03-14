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


def test_extract_squares() -> None:
    """Test that squares are correctly extracted from a board image."""
    # Create a test board image (512x512) with unique values for each square
    # This allows us to verify both position and orientation of each square
    board = np.zeros((512, 512), dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            # Fill each square with a unique pattern:
            # - Base value is i*8 + j (0-63) for position verification
            # - Add a gradient within each square for orientation verification
            square = np.zeros((64, 64), dtype=np.uint8)
            base_value = (i * 8 + j) * 2  # *2 to avoid overflow with gradient
            for x in range(64):
                for y in range(64):
                    # Add gradient: increases from top-left to bottom-right
                    gradient = (x + y) // 64
                    square[x, y] = base_value + gradient
            board[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = square

    # Test square extraction without flipping
    squares = ChessVision.extract_squares(board, flip=False)

    # Check shapes
    assert squares.shape == (64, 64, 64, 1)

    # Verify each square's position and orientation
    for idx in range(64):
        i, j = idx // 8, idx % 8
        square = squares[idx, :, :, 0]

        # Check base value (position verification)
        expected_base = (i * 8 + j) * 2
        assert square[0, 0] == expected_base, f"Square at position {idx} (i={i}, j={j}) has wrong base value"

        # Check gradient (orientation verification)
        assert square[63, 63] == expected_base + 1, f"Square at position {idx} (i={i}, j={j}) has wrong gradient"
        assert square[0, 63] == expected_base + 0, f"Square at position {idx} (i={i}, j={j}) has wrong left edge"
        assert square[63, 0] == expected_base + 0, f"Square at position {idx} (i={i}, j={j}) has wrong top edge"

    # Test square extraction with flipping
    squares_flipped = ChessVision.extract_squares(board, flip=True)

    # Check shapes
    assert squares_flipped.shape == (64, 64, 64, 1)

    # Verify each square's position and orientation in flipped board
    for idx in range(64):
        # Calculate original position that should now be at idx
        orig_i = 7 - (idx // 8)
        orig_j = 7 - (idx % 8)
        square = squares_flipped[idx, :, :, 0]

        # Check base value (position verification)
        expected_base = (orig_i * 8 + orig_j) * 2
        assert square[0, 0] == expected_base, f"Flipped square at position {idx} (orig_i={orig_i}, orig_j={orig_j}) has wrong base value"

        # Check gradient (orientation verification)
        assert square[63, 63] == expected_base + 1, f"Flipped square at position {idx} (orig_i={orig_i}, orig_j={orig_j}) has wrong gradient"
        assert square[0, 63] == expected_base + 0, f"Flipped square at position {idx} (orig_i={orig_i}, orig_j={orig_j}) has wrong left edge"
        assert square[63, 0] == expected_base + 0, f"Flipped square at position {idx} (orig_i={orig_i}, orig_j={orig_j}) has wrong top edge"


def test_get_square_names() -> None:
    """Test that square names are correctly generated."""
    # Test without flipping
    names = ChessVision._get_square_names(flip=False)

    # Check square names
    assert len(names) == 64
    assert names[0] == "a8"  # Top-left square
    assert names[7] == "h8"  # Top-right square
    assert names[56] == "a1"  # Bottom-left square
    assert names[63] == "h1"  # Bottom-right square

    # Test with flipping
    names_flipped = ChessVision._get_square_names(flip=True)

    # Check square names (should be reversed)
    assert len(names_flipped) == 64
    assert names_flipped[0] == "h1"  # Top-left square (was bottom-right)
    assert names_flipped[7] == "a1"  # Top-right square (was bottom-left)
    assert names_flipped[56] == "h8"  # Bottom-left square (was top-right)
    assert names_flipped[63] == "a8"  # Bottom-right square (was top-left)

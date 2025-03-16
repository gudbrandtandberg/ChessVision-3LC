"""Tests for evaluation metrics computation."""

from __future__ import annotations

import chess
import numpy as np

from chessvision import constants
from scripts.eval.evaluate import (
    TopKAccuracyResult,
    board_to_labels,
    compute_model_topk_accuracy,
)


def test_board_to_labels() -> None:
    """Test converting chess.Board to piece labels."""
    # Test starting position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    board = chess.Board(fen + " w KQkq - 0 1")
    labels = board_to_labels(board)

    # First 8 squares should be black pieces (8th rank)
    assert labels[:8] == ["r", "n", "b", "q", "k", "b", "n", "r"]
    # Next 8 squares should be black pawns (7th rank)
    assert labels[8:16] == ["p"] * 8
    # Middle ranks should be empty
    assert labels[16:48] == ["f"] * 32
    # Second to last rank should be white pawns (2nd rank)
    assert labels[48:56] == ["P"] * 8
    # Last rank should be white pieces (1st rank)
    assert labels[56:] == ["R", "N", "B", "Q", "K", "B", "N", "R"]

    # Test empty board
    board = chess.Board.empty()
    labels = board_to_labels(board)
    assert all(label == "f" for label in labels)

    # Test single piece
    board = chess.Board.empty()
    board.set_piece_at(chess.E4, chess.Piece.from_symbol("Q"))  # e4 in board coordinates
    labels = board_to_labels(board)
    # e4 is on the 4th rank from bottom, so it should be on the 5th rank from top
    e5_idx = 4 * 8 + 4  # 5th rank, 5th file in FEN coordinates
    assert labels[e5_idx] == "Q"
    assert sum(1 for label in labels if label != "f") == 1


def test_compute_top_k_accuracy() -> None:
    """Test top-k accuracy computation."""
    # Create predictions for a simple position:
    # - First 32 squares perfectly predicted
    # - Next 16 squares correct in top 2
    # - Last 16 squares correct in top 3
    predictions = np.zeros((64, 13), dtype=np.float32)  # 13 classes
    true_fen = "8/8/8/8/8/8/8/8"  # Empty board

    # Perfect predictions for first 32
    predictions[:32, constants.LABEL_INDICES["f"]] = 1.0

    # Top-2 predictions for next 16
    predictions[32:48, constants.LABEL_INDICES["p"]] = 1.0  # Wrong prediction
    predictions[32:48, constants.LABEL_INDICES["f"]] = 0.9  # Correct but lower confidence

    # Top-3 predictions for last 16
    predictions[48:, constants.LABEL_INDICES["P"]] = 1.0  # Wrong prediction
    predictions[48:, constants.LABEL_INDICES["p"]] = 0.9  # Wrong prediction
    predictions[48:, constants.LABEL_INDICES["f"]] = 0.8  # Correct but lowest confidence

    result = compute_model_topk_accuracy(predictions, true_fen, k=3)

    assert isinstance(result, TopKAccuracyResult)
    assert result.k == 3
    assert len(result.accuracies) == 3
    assert result.top_1 == 0.5  # 32/64 correct
    assert result.top_2 == 0.75  # 48/64 correct (32 from top-1 + 16 from top-2)
    assert result.top_3 == 1.0  # All correct (32 + 16 + 16)


def test_compute_top_k_accuracy_variable_k() -> None:
    """Test top-k accuracy with different k values."""
    # Test a simple position with white pawns on rank 2
    predictions = np.zeros((64, 13), dtype=np.float32)
    true_fen = "8/8/8/8/8/8/PPPPPPPP/8"

    # Perfect predictions for pawns on 2nd rank (indices 48-55 in FEN order)
    for i in range(48, 56):
        predictions[i, constants.LABEL_INDICES["P"]] = 1.0
    # Perfect predictions for empty squares
    for i in list(range(48)) + list(range(56, 64)):
        predictions[i, constants.LABEL_INDICES["f"]] = 1.0

    # Test k=1
    result_k1 = compute_model_topk_accuracy(predictions, true_fen, k=1)
    assert result_k1.k == 1
    assert len(result_k1.accuracies) == 1
    assert result_k1.top_1 == 1.0
    assert result_k1.top_2 == 0.0  # Not computed

    # Test k=5
    result_k5 = compute_model_topk_accuracy(predictions, true_fen, k=5)
    assert result_k5.k == 5
    assert len(result_k5.accuracies) == 5
    assert result_k5.top_1 == 1.0
    assert all(acc == 1.0 for acc in result_k5.accuracies)


def test_compute_position_metrics() -> None:
    """Test full position metrics computation."""
    # Test with a complex position that will trigger validation rules
    true_fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R"
    predictions = np.zeros((64, 13), dtype=np.float32)

    # Create predictions that will need validation
    board = chess.Board(true_fen + " w KQkq - 0 1")
    true_labels = board_to_labels(board)

    for square, label in enumerate(true_labels):
        if square < 8 or square >= 56:  # First and last ranks
            # Incorrectly predict pawns on first/last ranks to test validation
            predictions[square, constants.LABEL_INDICES["p" if square < 8 else "P"]] = 1.0
            # Add some probability for the correct piece as second choice
            predictions[square, constants.LABEL_INDICES[label]] = 0.8
        else:
            predictions[square, constants.LABEL_INDICES[label]] = 1.0

    result = compute_model_topk_accuracy(predictions, true_fen, k=3)

    # Check basic metrics
    assert result.k == 3
    assert len(result.accuracies) == 3
    assert result.top_1 < 1.0  # Should be less than 1.0 due to incorrect pawns
    assert result.top_2 > result.top_1  # Should improve with top-2


def test_compute_position_metrics_with_errors() -> None:
    """Test position metrics computation with imperfect predictions that need validation."""
    true_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"  # Starting position
    predictions = np.zeros((64, 13), dtype=np.float32)

    board = chess.Board(true_fen + " w KQkq - 0 1")
    true_labels = board_to_labels(board)

    # Create predictions with different confidence levels
    for square, label in enumerate(true_labels):
        if square < 8:  # First rank
            # Correct in third position
            predictions[square, constants.LABEL_INDICES["p"]] = 0.9  # Wrong (highest)
            predictions[square, constants.LABEL_INDICES["q"]] = 0.8  # Wrong (second)
            predictions[square, constants.LABEL_INDICES[label]] = 0.7  # Correct (third)
        elif square >= 56:  # Last rank
            # Correct in second position
            predictions[square, constants.LABEL_INDICES["P"]] = 0.9  # Wrong (highest)
            predictions[square, constants.LABEL_INDICES[label]] = 0.8  # Correct (second)
            predictions[square, constants.LABEL_INDICES["Q"]] = 0.7  # Wrong (third)
        else:
            # Correct in first position
            predictions[square, constants.LABEL_INDICES[label]] = 0.9  # Correct (highest)
            predictions[square, constants.LABEL_INDICES["f"]] = 0.8  # Wrong (second)
            predictions[square, constants.LABEL_INDICES["p"]] = 0.7  # Wrong (third)

    result = compute_model_topk_accuracy(predictions, true_fen, k=3)

    # Check metrics with precise expectations
    assert result.k == 3

    # Middle 48 squares correct in top-1, last rank 8 squares in top-2, first rank 8 squares in top-3
    expected_top1 = 40 / 64  # Only middle squares correct
    expected_top2 = 57 / 64  # Middle + last rank
    expected_top3 = 64 / 64  # All squares

    assert abs(result.top_1 - expected_top1) < 1e-6
    assert abs(result.top_2 - expected_top2) < 1e-6
    assert abs(result.top_3 - expected_top3) < 1e-6

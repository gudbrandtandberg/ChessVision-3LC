"""Tests for evaluation metrics computation."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chessvision.core import ChessVision
from chessvision.evaluate import (
    TopKAccuracyResult,
    board_to_labels,
    compute_position_metrics,
    compute_top_k_accuracy,
)


def test_board_to_labels():
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


def test_compute_top_k_accuracy():
    """Test top-k accuracy computation."""
    # Create predictions for a simple position:
    # - First 32 squares perfectly predicted
    # - Next 16 squares correct in top 2
    # - Last 16 squares correct in top 3
    predictions = np.zeros((64, 13))  # 13 classes
    true_labels = ["f"] * 64  # All empty squares

    # Perfect predictions for first 32
    predictions[:32, ChessVision.LABEL_INDICES["f"]] = 1.0

    # Top-2 predictions for next 16
    predictions[32:48, ChessVision.LABEL_INDICES["p"]] = 1.0  # Wrong prediction
    predictions[32:48, ChessVision.LABEL_INDICES["f"]] = 0.9  # Correct but lower confidence

    # Top-3 predictions for last 16
    predictions[48:, ChessVision.LABEL_INDICES["P"]] = 1.0  # Wrong prediction
    predictions[48:, ChessVision.LABEL_INDICES["p"]] = 0.9  # Wrong prediction
    predictions[48:, ChessVision.LABEL_INDICES["f"]] = 0.8  # Correct but lowest confidence

    result = compute_top_k_accuracy(predictions, true_labels, k=3)

    assert isinstance(result, TopKAccuracyResult)
    assert result.k == 3
    assert len(result.accuracies) == 3
    assert result.top_1 == 0.5  # 32/64 correct
    assert result.top_2 == 0.75  # 48/64 correct (32 from top-1 + 16 from top-2)
    assert result.top_3 == 1.0  # All correct (32 + 16 + 16)


def test_compute_top_k_accuracy_variable_k():
    """Test top-k accuracy with different k values."""
    # Test a simple position with white pawns on rank 2
    predictions = np.zeros((64, 13))
    true_fen = "8/8/8/8/8/8/PPPPPPPP/8"
    board = chess.Board(true_fen + " w KQkq - 0 1")
    true_labels = board_to_labels(board)

    # Perfect predictions for pawns on 2nd rank (indices 48-55 in FEN order)
    for i in range(48, 56):
        predictions[i, ChessVision.LABEL_INDICES["P"]] = 1.0
    # Perfect predictions for empty squares
    for i in list(range(48)) + list(range(56, 64)):
        predictions[i, ChessVision.LABEL_INDICES["f"]] = 1.0

    # Test k=1
    result_k1 = compute_top_k_accuracy(predictions, true_labels, k=1)
    assert result_k1.k == 1
    assert len(result_k1.accuracies) == 1
    assert result_k1.top_1 == 1.0
    assert result_k1.top_2 == 0.0  # Not computed

    # Test k=5
    result_k5 = compute_top_k_accuracy(predictions, true_labels, k=5)
    assert result_k5.k == 5
    assert len(result_k5.accuracies) == 5
    assert result_k5.top_1 == 1.0
    assert all(acc == 1.0 for acc in result_k5.accuracies)


def test_compute_position_metrics():
    """Test full position metrics computation."""
    # Test with a complex position
    true_fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R"
    predictions = np.zeros((64, 13))

    # Create perfect predictions for this position
    board = chess.Board(true_fen + " w KQkq - 0 1")
    true_labels = board_to_labels(board)

    for square, label in enumerate(true_labels):
        predictions[square, ChessVision.LABEL_INDICES[label]] = 1.0

    result = compute_position_metrics(predictions, true_fen, k=3)

    assert result.top_k_accuracy.k == 3
    assert result.top_k_accuracy.top_1 == 1.0  # All predictions correct
    assert len(result.true_labels) == 64
    assert len(result.predicted_labels) == 64

    # Verify specific pieces using FEN order indices
    e8_idx = 4  # Black king on e8 (5th square on 8th rank)
    e1_idx = 60  # White king on e1 (5th square on 1st rank)
    c4_idx = 34  # White bishop on c4 (3rd square on 4th rank)

    assert result.true_labels[e8_idx] == "k"  # Black king
    assert result.predicted_labels[e8_idx] == "k"
    assert result.true_labels[e1_idx] == "K"  # White king
    assert result.predicted_labels[e1_idx] == "K"
    assert result.true_labels[c4_idx] == "B"  # White bishop
    assert result.predicted_labels[c4_idx] == "B"


def test_compute_position_metrics_with_errors():
    """Test position metrics computation with imperfect predictions."""
    true_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"  # Starting position
    predictions = np.zeros((64, 13))

    # Make some intentional mistakes:
    # - Confuse knights with bishops (second most likely)
    # - Confuse pawns with empty squares
    board = chess.Board(true_fen + " w KQkq - 0 1")
    true_labels = board_to_labels(board)

    for square, label in enumerate(true_labels):
        if label in ["n", "N"]:  # Knights
            predictions[square, ChessVision.LABEL_INDICES["b" if label == "n" else "B"]] = 1.0  # Predict as bishop
            predictions[square, ChessVision.LABEL_INDICES[label]] = 0.9  # Correct as second choice
        elif label in ["p", "P"]:  # Pawns
            predictions[square, ChessVision.LABEL_INDICES["f"]] = 1.0  # Predict as empty
            predictions[square, ChessVision.LABEL_INDICES[label]] = 0.8  # Correct as third choice
        else:
            predictions[square, ChessVision.LABEL_INDICES[label]] = 1.0  # Perfect prediction

    result = compute_position_metrics(predictions, true_fen, k=3)

    # We should have:
    # - 44 perfect predictions (kings, queens, bishops, rooks, empty squares)
    # - 4 knights predicted as bishops (correct in top 2)
    # - 16 pawns predicted as empty (correct in top 3)
    assert result.top_k_accuracy.top_1 < 1.0  # Not all predictions correct
    assert result.top_k_accuracy.top_2 > result.top_k_accuracy.top_1  # Better with k=2
    assert result.top_k_accuracy.top_3 > result.top_k_accuracy.top_2  # Even better with k=3

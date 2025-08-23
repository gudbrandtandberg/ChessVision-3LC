#!/usr/bin/env python3
"""
ChessVision Pipeline Deep Dive

This script breaks down each step of the ChessVision pipeline, showing detailed
technical information at each stage. This is intended for developers and researchers
who want to understand the data flow and transformations.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.nn.functional import softmax

from chessvision import ChessVision, constants, utils


def print_tensor_info(name: str, tensor: torch.Tensor | np.ndarray) -> None:
    """Print shape and basic statistics of a tensor."""
    if isinstance(tensor, torch.Tensor):
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Device: {tensor.device}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        if tensor.dtype in [torch.float32, torch.float64]:
            print(f"  Mean: {tensor.mean():.3f}")
            print(f"  Std: {tensor.std():.3f}")
    elif isinstance(tensor, np.ndarray):
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        if tensor.dtype in [np.float32, np.float64]:
            print(f"  Mean: {tensor.mean():.3f}")
            print(f"  Std: {tensor.std():.3f}")


def print_board_ascii(fen: str) -> None:
    """Print chess board in ASCII format."""
    rows = fen.split("/")
    print("  +-----------------+")
    for row in rows:
        board_row = " |"
        for char in row:
            if char.isdigit():
                board_row += " ." * int(char)
            else:
                board_row += f" {char}"
        print(board_row + " |")
    print("  +-----------------+")


def main() -> None:
    flip = False  # Whether the board is shown from the white side
    threshold = 0.5  # Threshold for the board extraction

    device = utils.get_device()
    print(f"Using device: {device}")

    # 1. Initialize ChessVision
    print("\n=== Initializing ChessVision ===")
    chess_vision = ChessVision()

    # Print model information
    print("\nBoard Extractor (UNet):")
    print(f"Total parameters: {sum(p.numel() for p in chess_vision.board_extractor.parameters()):,}")

    print("\nPiece Classifier:")
    print(f"Total parameters: {sum(p.numel() for p in chess_vision.classifier.parameters()):,}")

    # 2. Load and Process Image
    print("\n=== Image Loading and Preprocessing ===")
    test_image_path = constants.DATA_ROOT / "test" / "raw" / "3cb7e9ca-0549-4072-a0ef-ae5ea82174e6.JPG"
    print(f"Loading image: {test_image_path}")

    original_img = cv2.imread(str(test_image_path))
    print_tensor_info("Original image", original_img)

    # 3. Process Image
    print("\n=== Processing Image ===")
    result = chess_vision.process_image(original_img, threshold=threshold, flip=flip)

    # Print detailed information about each stage
    print("\nBoard Extraction Results:")
    print(f"Processing time: {result.processing_time:.3f} seconds")

    if result.board_extraction.board_image is not None:
        print_tensor_info("Extracted board", result.board_extraction.board_image)
        print_tensor_info("Binary mask", result.board_extraction.binary_mask)

        if result.position:
            print("\nPosition Analysis:")
            print(f"FEN: {result.position.fen}")

            print("\nPredicted Position:")
            print_board_ascii(result.position.fen)

            # Show detailed predictions for first row
            print("\nDetailed First Row Analysis:")
            first_row_indices = range(0, 8)  # a8 to h8
            for i in first_row_indices:
                square_name = result.position.square_names[i]
                predictions = result.position.model_probabilities[i]
                probabilities = softmax(torch.tensor(predictions), dim=0)
                top_k = torch.topk(probabilities, k=3)

                print(f"\n{square_name}:")
                for prob, idx in zip(top_k.values, top_k.indices):
                    piece = constants.LABEL_NAMES[idx]
                    print(f"  {piece:2}: {prob:.2%}")
    else:
        print("No board detected in image!")


if __name__ == "__main__":
    main()

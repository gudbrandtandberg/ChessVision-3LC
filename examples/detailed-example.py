#!/usr/bin/env python3
"""
ChessVision Pipeline Deep Dive

This script breaks down each step of the ChessVision pipeline, showing detailed
technical information at each stage. This is intended for developers and researchers
who want to understand the data flow and transformations.
"""

import io
from pathlib import Path

import cv2
import torch

from chessvision.board_extraction.train_unet import load_checkpoint as load_extractor_checkpoint
from chessvision.piece_classification.train_classifier import get_classifier_model
from chessvision.piece_classification.train_classifier import load_checkpoint as load_classifier_checkpoint
from chessvision.predict.classify_board import classify_board
from chessvision.predict.extract_board import extract_board
from chessvision.pytorch_unet.unet.unet_model import UNet
from chessvision.utils import DATA_ROOT, INPUT_SIZE, best_classifier_weights, best_extractor_weights, get_device


def print_tensor_info(name, tensor):
    """Print shape and basic statistics of a tensor."""
    if isinstance(tensor, torch.Tensor):
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Device: {tensor.device}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    elif isinstance(tensor, np.ndarray):
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")


def main():
    device = get_device()
    print(f"Using device: {device}")

    # 1. Load Models
    print("\n=== Loading Models ===")

    # Load board extractor (UNet)
    print("\nInitializing UNet for board extraction...")
    extractor = UNet(n_channels=3, n_classes=1)
    extractor = extractor.to(memory_format=torch.channels_last)
    extractor = load_extractor_checkpoint(extractor, best_extractor_weights)
    extractor.eval()
    extractor.to(device)
    print(f"Total parameters: {sum(p.numel() for p in extractor.parameters()):,}")

    # Load classifier
    print("\nInitializing classifier...")
    classifier = get_classifier_model()
    classifier, _, _, _ = load_classifier_checkpoint(classifier, None, best_classifier_weights)
    classifier.eval()
    classifier.to(device)
    print(f"Total parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    # 2. Load and Process Image
    print("\n=== Image Loading and Preprocessing ===")
    test_image_path = Path(DATA_ROOT) / "test" / "raw" / "3cb7e9ca-0549-4072-a0ef-ae5ea82174e6.JPG"
    print(f"Loading image: {test_image_path}")

    original_img = cv2.imread(str(test_image_path))
    print(f"Original image shape: {original_img.shape}")

    comp_image = cv2.resize(original_img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    print(f"Resized image shape: {comp_image.shape}")

    # 3. Board Extraction
    print("\n=== Board Extraction ===")
    try:
        board_img, mask = extract_board(comp_image, original_img, extractor, threshold=0.3)
        if board_img is None:
            print("Failed to extract board!")
            return

        print(f"Segmentation mask shape: {mask.shape}")
        print(f"Extracted board shape: {board_img.shape}")

    except Exception as e:
        print(f"Error during board extraction: {e}")
        return

    # 4. Square Classification
    print("\n=== Square Classification ===")
    fen, predictions, chessboard, squares, names = classify_board(board_img, classifier)

    print(f"Number of squares extracted: {len(squares)}")
    print("\nFirst row predictions:")
    for i, (name, pred) in enumerate(zip(names[:8], predictions[:8])):
        confidence = torch.softmax(torch.tensor(pred), dim=0).max().item()
        print(f"  Square {chr(97 + i)}8: {name:12} (confidence: {confidence:.2%})")

    # 5. Final Results
    print("\n=== Final Results ===")
    print(f"Generated FEN: {fen}")

    # Print board in ASCII
    print("\nDetected position:")
    rows = fen.split("/")[:-1]  # Exclude the last part with move information
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


if __name__ == "__main__":
    main()

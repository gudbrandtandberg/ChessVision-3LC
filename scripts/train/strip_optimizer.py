#!/usr/bin/env python3
"""Script to strip optimizer state from classifier checkpoints."""

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def strip_optimizer(checkpoint_path: str, output_path: str | None = None) -> None:
    """Strip optimizer state from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file
        output_path: Optional path to save stripped checkpoint. If None, overwrites input file.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        logger.warning(f"Checkpoint at {checkpoint_path} is not a dictionary, skipping...")
        return

    # Create new checkpoint with only model state
    if "model_state_dict" in checkpoint:
        stripped = {
            "model_state_dict": checkpoint["model_state_dict"],
            "metadata": checkpoint.get("metadata", {}),
        }
    elif "state_dict" in checkpoint:
        stripped = {
            "state_dict": checkpoint["state_dict"],
            "metadata": checkpoint.get("metadata", {}),
        }
    else:
        logger.warning(f"Checkpoint at {checkpoint_path} has unexpected format, skipping...")
        return

    # Save stripped checkpoint
    output_path = output_path or checkpoint_path
    torch.save(stripped, output_path)
    logger.info(f"Saved stripped checkpoint to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip optimizer state from classifier checkpoints")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="C:/Project/ChessVision-3LC/weights/best_extractor.pth",
        help="Path to checkpoint file or directory",
    )
    parser.add_argument("--output-dir", type=str, help="Optional output directory for stripped checkpoints")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    checkpoint_path = Path(args.checkpoint_path)
    if checkpoint_path.is_dir():
        # Process all .pt and .pth files in directory
        for ext in ["*.pt", "*.pth"]:
            for file in checkpoint_path.glob(ext):
                if args.output_dir:
                    output_path = Path(args.output_dir) / file.name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    output_path = file
                strip_optimizer(str(file), str(output_path))
    else:
        # Process single file
        if args.output_dir:
            output_path = Path(args.output_dir) / checkpoint_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = checkpoint_path
        strip_optimizer(str(checkpoint_path), str(output_path))


if __name__ == "__main__":
    main()

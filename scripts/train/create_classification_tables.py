from __future__ import annotations

import argparse
import logging
from pathlib import Path

import tlc
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL.Image import Image

from chessvision.core import ChessVision

logger = logging.getLogger(__name__)

DATASET_ROOT = f"{ChessVision.DATA_ROOT}/squares"
tlc.register_url_alias("CHESSPIECES_DATASET_ROOT", DATASET_ROOT)

TRAIN_DATASET_PATH = DATASET_ROOT + "/training"
VAL_DATASET_PATH = DATASET_ROOT + "/validation"

TRAIN_DATASET_NAME = "chesspieces-train"
VAL_DATASET_NAME = "chesspieces-val"

# Define transforms
train_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=0),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.564], [0.246]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.564], [0.246]),
    ]
)


def train_map(sample: tuple[Image, int]) -> tuple[torch.Tensor, int]:
    return train_transforms(sample[0]), sample[1]


def val_map(sample: tuple[Image, int]) -> tuple[torch.Tensor, int]:
    return val_transforms(sample[0]), sample[1]


def create_tables(project_name: str = "chessvision-classification") -> tuple[tlc.Table, tlc.Table]:
    """Create initial train/val tables for piece classification from raw data.

    Args:
        project_name: Name of the TLC project

    Returns:
        Tuple of (train_table, val_table)
    """
    logger.info("Creating piece classification tables...")
    logger.info(f"Using data from {DATASET_ROOT}")

    # Verify paths exist
    train_path = Path(TRAIN_DATASET_PATH)
    val_path = Path(VAL_DATASET_PATH)

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found at {val_path}")

    # Create datasets
    train_dataset = datasets.ImageFolder(TRAIN_DATASET_PATH)
    val_dataset = datasets.ImageFolder(VAL_DATASET_PATH)

    logger.info(f"Found {len(train_dataset)} training images")
    logger.info(f"Found {len(val_dataset)} validation images")

    # Define table structure
    sample_structure = (tlc.PILImage("image"), tlc.CategoricalLabel("label", classes=ChessVision.LABEL_NAMES))

    # Create tables
    tlc_train_dataset = (
        tlc.Table.from_torch_dataset(
            dataset=train_dataset,
            dataset_name=TRAIN_DATASET_NAME,
            table_name="train",
            structure=sample_structure,
            project_name=project_name,
        )
        .map(train_map)
        .map_collect_metrics(val_map)
        .revision()
    )

    tlc_val_dataset = (
        tlc.Table.from_torch_dataset(
            dataset=val_dataset,
            dataset_name=VAL_DATASET_NAME,
            table_name="val",
            structure=sample_structure,
            project_name=project_name,
        )
        .map(val_map)
        .revision()
    )

    logger.info(f"Created training table: {tlc_train_dataset.url}")
    logger.info(f"Created validation table: {tlc_val_dataset.url}")

    return tlc_train_dataset, tlc_val_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-name", type=str, default="chessvision-classification")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    create_tables(args.project_name)


if __name__ == "__main__":
    main()

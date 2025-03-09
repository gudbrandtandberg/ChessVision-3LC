from __future__ import annotations

import logging

import tlc
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL.Image import Image

from chessvision.core import ChessVision

from . import config

logger = logging.getLogger(__name__)

# Define transforms
train_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=0),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.564], [0.246]),
    ],
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.564], [0.246]),
    ],
)


def train_map(sample: tuple[Image, int]) -> tuple[torch.Tensor, int]:
    return train_transforms(sample[0]), sample[1]


def val_map(sample: tuple[Image, int]) -> tuple[torch.Tensor, int]:
    return val_transforms(sample[0]), sample[1]


def create_tables() -> dict[str, tlc.Table]:
    """Create initial train/val tables for piece classification from raw data."""
    logger.info("Creating piece classification tables...")
    logger.info(f"Using data from {config.PIECE_CLASSIFICATION_ROOT}")

    # Verify paths exist
    for path in config.PIECE_CLASSIFICATION_PATHS.values():
        if not path.exists():
            raise FileNotFoundError(path)

    # Create datasets
    train_dataset = datasets.ImageFolder(str(config.PIECE_CLASSIFICATION_PATHS["train"]))
    val_dataset = datasets.ImageFolder(str(config.PIECE_CLASSIFICATION_PATHS["val"]))

    logger.info(f"Found {len(train_dataset)} training images")
    logger.info(f"Found {len(val_dataset)} validation images")

    # Define table structure
    sample_structure = (tlc.PILImage("image"), tlc.CategoricalLabel("label", classes=ChessVision.LABEL_NAMES))

    # Create tables
    tables = {}
    tables["train"] = tlc.Table.from_torch_dataset(
        dataset=train_dataset,
        dataset_name=config.PIECE_CLASSIFICATION_DATASETS["train"],
        table_name="train",
        structure=sample_structure,
        project_name=config.PIECE_CLASSIFICATION_PROJECT,
    )

    tables["val"] = tlc.Table.from_torch_dataset(
        dataset=val_dataset,
        dataset_name=config.PIECE_CLASSIFICATION_DATASETS["val"],
        table_name="val",
        structure=sample_structure,
        project_name=config.PIECE_CLASSIFICATION_PROJECT,
    )

    logger.info(f"Created training table: {tables['train'].url}")
    logger.info(f"Created validation table: {tables['val'].url}")

    return tables


def get_or_create_tables(
    train_table_name: str,
    val_table_name: str,
) -> dict[str, tlc.Table]:
    """Get existing tables or create new ones if they don't exist."""
    try:
        tables = {}
        tables["train"] = tlc.Table.from_names(
            table_name=train_table_name,
            dataset_name=config.PIECE_CLASSIFICATION_DATASETS["train"],
            project_name=config.PIECE_CLASSIFICATION_PROJECT,
        )

        tables["val"] = tlc.Table.from_names(
            table_name=val_table_name,
            dataset_name=config.PIECE_CLASSIFICATION_DATASETS["val"],
            project_name=config.PIECE_CLASSIFICATION_PROJECT,
        )

        logger.info("Using existing tables:")
        logger.info(f"Training: {tables['train'].url}")
        logger.info(f"Validation: {tables['val'].url}")

    except Exception:
        logger.info("Tables not found, creating new ones...")
        tables = create_tables()

    return tables


if __name__ == "__main__":
    tables = get_or_create_tables(
        train_table_name=config.INITIAL_TABLE_NAME,
        val_table_name=config.INITIAL_TABLE_NAME,
    )
    logger.info(tables)

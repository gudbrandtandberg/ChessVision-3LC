from __future__ import annotations

import logging

import tlc
import torchvision.datasets as datasets

from chessvision import constants
from scripts.train import config

logger = logging.getLogger(__name__)


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
    sample_structure = (
        tlc.PILImage("image"),
        tlc.CategoricalLabel("label", classes=constants.LABEL_NAMES),
    )

    # Create tables
    tables = {}
    tables["train"] = tlc.Table.from_torch_dataset(
        dataset=train_dataset,
        structure=sample_structure,
        table_name=config.INITIAL_TABLE_NAME,
        dataset_name=config.PIECE_CLASSIFICATION_DATASETS["train"],
        project_name=config.PIECE_CLASSIFICATION_PROJECT,
    )

    tables["val"] = tlc.Table.from_torch_dataset(
        dataset=val_dataset,
        structure=sample_structure,
        table_name=config.INITIAL_TABLE_NAME,
        dataset_name=config.PIECE_CLASSIFICATION_DATASETS["val"],
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

    except FileNotFoundError:
        if train_table_name == config.INITIAL_TABLE_NAME and val_table_name == config.INITIAL_TABLE_NAME:
            logger.info("Initial tables not found, creating new ones...")
            tables = create_tables()
        else:
            logger.warning(f"Could not find tables: {train_table_name} and {val_table_name}")
            raise FileNotFoundError(train_table_name) from None

    return tables


if __name__ == "__main__":
    tables = get_or_create_tables(
        train_table_name=config.INITIAL_TABLE_NAME,
        val_table_name=config.INITIAL_TABLE_NAME,
    )
    print(tables)

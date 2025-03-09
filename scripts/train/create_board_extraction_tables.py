import logging

import tlc
import torch
from torch.utils.data import random_split

from chessvision.core import ChessVision
from chessvision.pytorch_unet.utils.data_loading import BasicDataset
from scripts.train import config

logger = logging.getLogger(__name__)


def create_tables() -> dict[str, tlc.Table]:
    """Create initial train/val tables for board extraction from raw data.

    Args:
        project_name: Name of the TLC project
        val_percent: Percentage of data to use for validation

    Returns:
        Dictionary containing 'train' and 'val' tables
    """
    logger.info("Creating board extraction tables...")
    logger.info(f"Using data from {config.BOARD_EXTRACTION_ROOT}")

    # Verify paths exist
    if not config.BOARD_EXTRACTION_PATHS["images"].exists():
        raise FileNotFoundError(config.BOARD_EXTRACTION_PATHS["images"])
    if not config.BOARD_EXTRACTION_PATHS["masks"].exists():
        raise FileNotFoundError(config.BOARD_EXTRACTION_PATHS["masks"])

    # Create dataset
    dataset = BasicDataset(
        images_dir=config.BOARD_EXTRACTION_PATHS["images"].as_posix(),
        mask_dir=config.BOARD_EXTRACTION_PATHS["masks"].as_posix(),
        scale=1.0,
    )
    logger.info(f"Found {len(dataset)} total images")

    # Split into train / validation partitions
    n_val = int(len(dataset) * config.VAL_SPLIT_PERCENT)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )

    logger.info(f"Split into {n_train} training and {n_val} validation images")

    sample_structure = {
        "image": tlc.PILImage("image"),
        "mask": tlc.SegmentationPILImage(
            "mask",
            classes=ChessVision.SEGMENTATION_MAP,
        ),
    }

    tables = {}
    tables["train"] = tlc.Table.from_torch_dataset(
        dataset=train_set,
        dataset_name=config.BOARD_EXTRACTION_DATASETS["train"],
        table_name="initial",
        structure=sample_structure,
        project_name=config.BOARD_EXTRACTION_PROJECT,
        if_exists="reuse",
    )

    tables["val"] = tlc.Table.from_torch_dataset(
        dataset=val_set,
        dataset_name=config.BOARD_EXTRACTION_DATASETS["val"],
        table_name="initial",
        structure=sample_structure,
        project_name=config.BOARD_EXTRACTION_PROJECT,
        if_exists="reuse",
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
        # Try to get existing tables
        tables = {}
        tables["train"] = tlc.Table.from_names(
            table_name=train_table_name,
            dataset_name=config.BOARD_EXTRACTION_DATASETS["train"],
            project_name=config.BOARD_EXTRACTION_PROJECT,
        )

        tables["val"] = tlc.Table.from_names(
            table_name=val_table_name,
            dataset_name=config.BOARD_EXTRACTION_DATASETS["val"],
            project_name=config.BOARD_EXTRACTION_PROJECT,
        )

    except FileNotFoundError:
        logger.info("Tables not found, creating new ones...")
        tables = create_tables()

    return tables


if __name__ == "__main__":
    tables = get_or_create_tables(
        train_table_name=config.INITIAL_TABLE_NAME,
        val_table_name=config.INITIAL_TABLE_NAME,
    )
    print(tables)

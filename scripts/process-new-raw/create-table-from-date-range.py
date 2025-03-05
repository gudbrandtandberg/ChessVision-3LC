#!/usr/bin/env python
"""
Create a TLC table from raw chess images.

This script creates a TLC table from a folder of raw chess images.
The table name is derived from the folder name.

Example usage:
    python scripts/process-new-raw/create-table-from-date-range.py --input_folder data/new_raw/2024-11-01-2024-11-01
"""

import argparse
import logging
from pathlib import Path

import tlc
import tqdm
from PIL import Image

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("create_tlc_table")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a TLC table from raw chess images.",
    )
    parser.add_argument(
        "--input_folder",
        help="Folder containing the raw images",
        required=True,
    )
    parser.add_argument(
        "--project_name",
        help="TLC project name",
        default="chessvision-new-raw",
    )
    parser.add_argument(
        "--dataset_name",
        help="TLC dataset name",
        default="chessvision-new-raw",
    )
    return parser.parse_args()


def main():
    """Create a TLC table from images in the input folder."""
    args = parse_arguments()

    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        return

    # Use the folder name as the table name, replacing hyphens with underscores
    table_name = input_folder.name.replace("-", "_").lower()

    logger.info(f"Creating table '{table_name}' from folder: {input_folder}")
    logger.info(f"Project: {args.project_name}, Dataset: {args.dataset_name}")

    # Create table writer
    table_writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=args.dataset_name,
        project_name=args.project_name,
        column_schemas={
            "image": tlc.PILImage("image"),
        },
        if_exists="overwrite",
    )

    # Add images to table
    image_count = 0
    image_files = list(input_folder.glob("*.JPG"))
    logger.info(f"Found {len(image_files)} JPG files in folder")

    for file in tqdm.tqdm(image_files, desc="Adding images to table"):
        if not file.is_file():
            continue

        try:
            pil_img = Image.open(file)
            table_writer.add_row(
                {
                    "image": pil_img,
                }
            )
            image_count += 1
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")

    logger.info(f"Added {image_count} images to table")

    # Finalize table
    table = table_writer.finalize()
    logger.info(f"Table created successfully: {table.url}")

    # Print the table URL for easy access
    logger.info(f"Table URL: {table.url}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Download raw chess images from S3 bucket for a specific date or date range.

This script downloads images from an S3 bucket organized by date hierarchy
(raw-uploads/YYYY/MM/DD/) and saves them to a local folder.
"""

import argparse
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import boto3
from tqdm import tqdm

from chessvision.utils import DATA_ROOT

# Configure logging to reduce noise from AWS libraries
logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("s3transfer").setLevel(logging.CRITICAL)

# Set up logger for this script
logger = logging.getLogger("download_raw_data")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download images from S3 bucket for a specific date or date range.",
    )
    parser.add_argument(
        "--bucket",
        help="The name of the S3 bucket.",
        default="chessvision-bucket",
    )
    parser.add_argument(
        "--start_date",
        help="The start date in YYYY-MM-DD format.",
        default="2024-11-1",
    )
    parser.add_argument(
        "--end_date",
        help="The end date in YYYY-MM-DD format. If not provided, only the start_date will be used.",
        default="2024-11-1",
    )
    parser.add_argument(
        "--output_folder",
        help="The folder where the images will be downloaded.",
        default=DATA_ROOT + "/new_raw",
    )
    parser.add_argument(
        "--boto_output",
        action="store_true",
        help="Enable boto3 output for debugging.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List files that would be downloaded without actually downloading them.",
    )
    return parser.parse_args()


def download_images_by_prefix(
    s3_client,
    bucket: str,
    prefix: str,
    output_folder: Path,
    boto_output: bool = False,
    dry_run: bool = False,
) -> int:
    """
    Download all images with the given prefix from S3 bucket.

    Args:
        s3_client: Initialized boto3 S3 client
        bucket: Name of the S3 bucket
        prefix: Prefix to filter objects (e.g., 'raw-uploads/2023/11/01/')
        output_folder: Local folder to save downloaded images
        boto_output: Whether to print detailed boto3 output
        dry_run: If True, only list files without downloading

    Returns:
        Number of files downloaded or listed
    """
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    count = 0

    if "Contents" not in response:
        logger.info(f"No files found with prefix: {prefix}")
        return 0

    files = response["Contents"]
    logger.info(f"Found {len(files)} files with prefix: {prefix}")

    for obj in tqdm(files, desc=f"Processing {prefix}"):
        key = obj["Key"]
        # Use the full date range for the folder name
        file_name = output_folder / os.path.basename(key)

        if boto_output:
            logger.info(f"{'Would download' if dry_run else 'Downloading'} {key} to {file_name}")

        if not dry_run:
            s3_client.download_file(bucket, key, str(file_name))
            count += 1
        else:
            count += 1

    return count


def main():
    """Main function to download images from S3."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    args = parse_arguments()

    bucket = args.bucket
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else start_date
    output_folder = Path(args.output_folder)
    boto_output = args.boto_output
    dry_run = args.dry_run
    date_folder = f"{start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}"
    output_folder = output_folder / date_folder
    logger.info(f"Downloading images from bucket: {bucket}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Dry run: {dry_run}")

    s3_client = boto3.client("s3")

    if not dry_run and not output_folder.exists():
        output_folder.mkdir(parents=True)
        logger.info(f"Created output directory: {output_folder}")

    total_files = 0
    current_date = start_date

    while current_date <= end_date:
        prefix = f"raw-uploads/{current_date.year}/{current_date.month}/{current_date.day}/"
        files_processed = download_images_by_prefix(
            s3_client,
            bucket,
            prefix,
            output_folder,
            boto_output,
            dry_run,
        )
        total_files += files_processed
        current_date += timedelta(days=1)

    action = "Listed" if dry_run else "Downloaded"
    logger.info(f"{action} {total_files} files in total")


if __name__ == "__main__":
    main()

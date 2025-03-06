#!/usr/bin/env python
"""
Chess Vision Processing Pipeline

This script orchestrates the complete workflow for processing raw chess images:
1. Download raw images from S3
2. Create a 3LC table from the downloaded images
3. Enrich the table with board extraction metrics

The script can be run as a standalone command or imported and used programmatically.

Example usage:
    # Run the complete pipeline
    python process_pipeline.py --start_date 2024-11-01 --end_date 2024-11-02

    # Run only specific steps
    python process_pipeline.py --start_date 2024-11-01 --skip_download --skip_enrich
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import boto3
import cv2
import numpy as np
import tlc
import torch
import torchvision.transforms.v2 as v2
from PIL import Image
from tqdm import tqdm

from chessvision.predict.classify_board import classify_board
from chessvision.predict.classify_raw import load_board_extractor, load_classifier
from chessvision.predict.extract_board import BoardExtractor
from chessvision.test.test import save_svg
from chessvision.utils import BOARD_SIZE, DATA_ROOT, INPUT_SIZE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("chessvision_pipeline")


def download_raw_data(
    bucket: str,
    start_date: datetime.date,
    end_date: datetime.date,
    output_folder: Optional[Path] = None,
    dry_run: bool = False,
) -> Path:
    """
    Download raw chess images from S3 bucket for a date range.

    Args:
        bucket: S3 bucket name
        start_date: Start date to download
        end_date: End date to download
        output_folder: Folder to save images (if None, creates one based on dates)
        dry_run: If True, only list files without downloading

    Returns:
        Path to the folder containing downloaded images
    """
    logger.info(f"Downloading images from bucket: {bucket}")
    logger.info(f"Date range: {start_date} to {end_date}")
    date_folder = f"{start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}"

    # Create output folder if not specified
    if output_folder is None:
        output_folder = Path(DATA_ROOT) / "new_raw" / date_folder
    else:
        output_folder = output_folder / date_folder

    logger.info(f"Output folder: {output_folder}")

    # Initialize S3 client
    s3_client = boto3.client("s3")

    if not dry_run and not output_folder.exists():
        output_folder.mkdir(parents=True)
        logger.info(f"Created output directory: {output_folder}")

    total_files = 0
    current_date = start_date
    while current_date <= end_date:
        prefix = f"raw-uploads/{current_date.year}/{current_date.month}/{current_date.day}/"

        # Initialize pagination
        continuation_token = None
        while True:
            # List objects with pagination
            list_kwargs = {
                "Bucket": bucket,
                "Prefix": prefix,
                "MaxKeys": 1000,  # AWS default is 1000, but being explicit
            }
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token

            response = s3_client.list_objects_v2(**list_kwargs)

            if "Contents" not in response:
                logger.info(f"No files found with prefix: {prefix}")
                break

            files = response["Contents"]
            logger.info(f"Processing {len(files)} files with prefix: {prefix}")

            for obj in tqdm(files, desc=f"Processing {prefix}"):
                key = obj["Key"]
                file_name = output_folder / Path(key).name

                if not dry_run:
                    s3_client.download_file(bucket, key, str(file_name))
                total_files += 1

            # Check if there are more files to fetch
            if not response.get("IsTruncated"):  # No more files
                break

            continuation_token = response.get("NextContinuationToken")
            if not continuation_token:  # Shouldn't happen if IsTruncated is True
                break

        current_date += timedelta(days=1)

    action = "Listed" if dry_run else "Downloaded"
    logger.info(f"{action} {total_files} files in total")

    return output_folder


def create_tlc_table(
    input_folder: Path,
    project_name: str = "chessvision-new-raw",
    dataset_name: str = "chessvision-new-raw",
) -> tlc.Table:
    """
    Create a 3LC table from images in the input folder.

    Args:
        input_folder: Path to folder containing images
        project_name: 3LC project name
        dataset_name: 3LC dataset name

    Returns:
        Created 3LC table
    """
    logger.info(f"Creating table from folder: {input_folder}")

    # Use the folder name as the table name
    table_name = input_folder.name.lower()

    logger.info(f"Table name: {table_name}")
    logger.info(f"Project: {project_name}, Dataset: {dataset_name}")

    # Create table writer
    table_writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=dataset_name,
        project_name=project_name,
        column_schemas={
            "image": tlc.PILImage("image"),
        },
        if_exists="overwrite",
    )

    # Add images to table
    image_count = 0
    image_files = list(input_folder.glob("*.JPG"))
    logger.info(f"Found {len(image_files)} JPG files in folder")

    for file in tqdm(image_files, desc="Adding images to table"):
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
    logger.info(f"Table created: {table.url}")

    return table


def enrich_tlc_table(
    table: tlc.Table,
    run_name: Optional[str] = None,
    threshold: float = 0.3,
    embedding_layer: int = 52,
) -> str:
    """
    Enrich a 3LC table with chess board extraction metrics.

    Args:
        table: 3LC table to enrich
        run_name: Name for the 3LC run
        threshold: Threshold for board extraction
        embedding_layer: Layer to extract embeddings from

    Returns:
        URL of the created run
    """
    # Generate run name if not provided
    if run_name is None:
        run_name = f"enrich-{table.name}"

    logger.info(f"Enriching table: {table.name}")
    logger.info(f"Run name: {run_name}")

    # Initialize 3LC run
    run = tlc.init(table.project_name, run_name)

    # Load model and create board extractor
    logger.info("Loading board extraction model")
    model = load_board_extractor()
    board_extractor = BoardExtractor(model)

    # Define preprocessing function
    def preprocess_image(sample):
        transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.ConvertImageDtype(torch.float32),
            ]
        )
        return transforms(sample["image"])

    # Preprocess images
    logger.info("Preprocessing images")
    table.map(preprocess_image)

    # Define metrics collector
    def custom_metrics_collector(batch, predictor_output):
        """Extract boards and collect metrics from batch of images."""
        logits = predictor_output.forward
        probs = torch.sigmoid(logits.squeeze(1))
        confidence = torch.abs(probs - 0.5) * 2

        batch_size = confidence.shape[0]
        confidences = []
        quadrangle_scores = []
        mask_completeness_scores = []
        prob_distribution_scores = []
        extracted_boards = []
        logit_images = []
        rendered_boards = []

        for i in range(batch_size):
            # Original confidence calculation
            k = int(confidence[i].numel() * 0.25)
            top_k_confidence = torch.topk(confidence[i].flatten(), k).values
            avg_confidence = float(top_k_confidence.mean())
            confidences.append(avg_confidence)

            # Get original image
            orig_image = (batch[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # Extract board
            result = board_extractor.extract_board(
                cv2.resize(orig_image, INPUT_SIZE, interpolation=cv2.INTER_AREA),
                orig_image,
                threshold=threshold,
            )

            # Calculate additional metrics
            quad_score = quadrangle_regularity(result.quadrangle)
            completeness_score = mask_completeness(result.probability_mask)
            distribution_score = probability_distribution(result.probability_mask)

            quadrangle_scores.append(quad_score)
            mask_completeness_scores.append(completeness_score)
            prob_distribution_scores.append(distribution_score)

            # Process results
            if result.board_image is not None:
                pil_board = Image.fromarray(result.board_image)
                extracted_boards.append(pil_board)
                sq_model = load_classifier()
                _, _, chessboard, _, _ = classify_board(result.board_image, sq_model, flip=False)
                svg_url = Path((run.bulk_data_url / "rendered_board" / "board.png").create_unique().to_str())
                svg_url.parent.mkdir(parents=True, exist_ok=True)
                save_svg(chessboard, svg_url)
                rendered_boards.append(Image.open(svg_url))

            else:
                black_image = Image.new("L", BOARD_SIZE, color=0)
                extracted_boards.append(black_image)
                rendered_boards.append(black_image)

            # Process mask
            logit_np = result.probability_mask.copy()
            min_val, max_val = logit_np.min(), logit_np.max()
            if max_val > min_val:
                logit_np = (logit_np - min_val) / (max_val - min_val)
            logit_np = (logit_np * 255).astype(np.uint8)
            logit_image = Image.fromarray(logit_np, mode="L")
            logit_images.append(logit_image)

        return {
            "confidence": confidences,
            "quadrangle_score": quadrangle_scores,
            "mask_completeness": mask_completeness_scores,
            "probability_distribution": prob_distribution_scores,
            "extracted_boards": extracted_boards,
            "rendered_boards": rendered_boards,
            "logit_images": logit_images,
        }

    # Define collector schemas
    custom_metrics_collector_schemas = {
        "confidence": tlc.Float("confidence"),
        "quadrangle_score": tlc.Float("quadrangle_score"),
        "mask_completeness": tlc.Float("mask_completeness"),
        "probability_distribution": tlc.Float("probability_distribution"),
        "extracted_boards": tlc.PILImage("extracted_boards"),
        "logit_images": tlc.PILImage("logit_images"),
        "rendered_boards": tlc.PILImage("rendered_boards"),
    }

    # Collect metrics
    logger.info("Collecting metrics and extracting boards")
    embedding_collector = tlc.EmbeddingsMetricsCollector([embedding_layer])

    tlc.collect_metrics(
        table,
        [
            tlc.FunctionalMetricsCollector(
                custom_metrics_collector,
                column_schemas=custom_metrics_collector_schemas,
                compute_aggregates=False,
            ),
            embedding_collector,
        ],
        tlc.Predictor(
            model,
            layers=[embedding_layer],
            preprocess_fn=v2.Resize((256, 256), antialias=True),
        ),
        collect_aggregates=False,
        dataloader_args={"batch_size": 1},
    )

    # Reduce embeddings
    logger.info("Reducing embeddings")
    run.reduce_embeddings_by_foreign_table_url(table.url)

    logger.info(f"Enrichment complete: {run.url}")
    return run.url


def probability_distribution(mask):
    """
    Analyze the distribution of probabilities in the mask.

    Args:
        mask: Probability mask from the model

    Returns:
        Score between 0-1 where higher values indicate a clearer decision boundary
    """
    # Calculate histogram of probability values
    hist, _ = np.histogram(mask.flatten(), bins=10, range=(0, 1))
    hist = hist / np.sum(hist)

    # Ideal distribution has values concentrated at 0 and 1
    # Calculate entropy (lower entropy means more concentrated distribution)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    max_entropy = -np.log2(1 / 10)  # Maximum entropy for 10 bins

    # Convert to score (1 - normalized entropy)
    return 1.0 - (entropy / max_entropy)


def mask_completeness(mask):
    """
    Measure how complete and solid the mask is.

    Args:
        mask: Probability mask from the model

    Returns:
        Score between 0-1 where 1 is a complete, solid mask
    """
    # Binarize the mask
    binary_mask = (mask > 0.5).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a filled mask from the largest contour
    filled_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filled_mask, [largest_contour], 0, 1, -1)

    # Calculate the ratio of the original mask area to the filled contour area
    original_area = np.sum(binary_mask)
    filled_area = np.sum(filled_mask)

    if filled_area == 0:
        return 0.0

    # Higher ratio means more complete mask
    return original_area / filled_area


def quadrangle_regularity(quadrangle):
    """
    Measure how close the quadrangle is to a perfect square.

    Args:
        quadrangle: np.ndarray of shape (4, 2) with corner coordinates

    Returns:
        Score between 0-1 where 1 is a perfect square
    """
    if quadrangle is None:
        return 0.0

    # Create a copy to avoid modifying input
    quadrangle = quadrangle.copy().squeeze(1)  # input is (4, 1, 2)

    # Calculate side lengths
    sides = []
    for i in range(4):
        next_i = (i + 1) % 4
        side_length = np.sqrt(((quadrangle[i] - quadrangle[next_i]) ** 2).sum())
        sides.append(side_length)

    # Calculate angles
    angles = []
    for i in range(4):
        prev_i = (i - 1) % 4
        next_i = (i + 1) % 4
        v1 = quadrangle[prev_i] - quadrangle[i]
        v2 = quadrangle[next_i] - quadrangle[i]
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        angle = np.arccos(dot / norm) if norm > 0 else 0
        angles.append(angle)

    # Ideal square has equal sides and 90-degree angles
    side_variance = np.std(sides) / np.mean(sides) if np.mean(sides) > 0 else 1.0
    angle_variance = np.std(angles) / (np.pi / 2)

    # Combine metrics (lower is better, so subtract from 1)
    return 1.0 - (side_variance * 0.5 + angle_variance * 0.5)


def run_pipeline(
    start_date: datetime.date,
    end_date: datetime.date,
    bucket: str = "chessvision-bucket",
    output_folder: Optional[Path] = None,
    project_name: str = "chessvision-new-raw",
    dataset_name: str = "chessvision-new-raw",
    run_name: Optional[str] = None,
    threshold: float = 0.3,
    embedding_layer: int = 52,
    skip_download: bool = False,
    skip_create_table: bool = False,
    skip_enrich: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Run the complete chess vision processing pipeline.

    Args:
        start_date: Start date to process
        end_date: End date to process
        bucket: S3 bucket name
        output_folder: Folder to save images (if None, creates one based on dates)
        project_name: 3LC project name
        dataset_name: 3LC dataset name
        run_name: Name for the 3LC run
        threshold: Threshold for board extraction
        embedding_layer: Layer to extract embeddings from
        skip_download: Skip the download step
        skip_create_table: Skip the table creation step
        skip_enrich: Skip the enrichment step
        dry_run: If True, only list files without downloading

    Returns:
        Dictionary with results from each step
    """
    results = {}

    # Step 1: Download raw data
    if not skip_download:
        logger.info("=== Step 1: Downloading raw data ===")
        download_folder = download_raw_data(
            bucket=bucket,
            start_date=start_date,
            end_date=end_date,
            output_folder=output_folder,
            dry_run=dry_run,
        )
        results["download_folder"] = download_folder
    else:
        logger.info("Skipping download step")
        if output_folder is None:
            date_folder = f"{start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}"
            download_folder = Path(DATA_ROOT) / "new_raw" / date_folder
        else:
            download_folder = output_folder
        results["download_folder"] = download_folder

    # Step 2: Create 3LC table
    if not skip_create_table:
        logger.info("=== Step 2: Creating 3LC table ===")
        table = create_tlc_table(
            input_folder=download_folder,
            project_name=project_name,
            dataset_name=dataset_name,
        )
        results["table"] = table
    else:
        logger.info("Skipping table creation step")
        table_name = download_folder.name.lower()
        table = tlc.Table.from_names(
            table_name,
            dataset_name,
            project_name,
        )
        results["table"] = table

    # Step 3: Enrich 3LC table
    if not skip_enrich:
        logger.info("=== Step 3: Enriching 3LC table ===")
        run_url = enrich_tlc_table(
            table=table,
            run_name=run_name,
            threshold=threshold,
            embedding_layer=embedding_layer,
        )
        results["run_url"] = run_url
    else:
        logger.info("Skipping enrichment step")

    return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the complete chess vision processing pipeline.",
    )
    parser.add_argument(
        "--start_date",
        help="Start date in YYYY-MM-DD format",
        required=True,
    )
    parser.add_argument(
        "--end_date",
        help="End date in YYYY-MM-DD format (defaults to start_date)",
        default=None,
    )
    parser.add_argument(
        "--bucket",
        help="S3 bucket name",
        default="chessvision-bucket",
    )
    parser.add_argument(
        "--output_folder",
        help="Folder to save images",
        default=None,
    )
    parser.add_argument(
        "--project_name",
        help="3LC project name",
        default="chessvision-new-raw",
    )
    parser.add_argument(
        "--dataset_name",
        help="3LC dataset name",
        default="chessvision-new-raw",
    )
    parser.add_argument(
        "--run_name",
        help="Name for the 3LC run",
        default=None,
    )
    parser.add_argument(
        "--threshold",
        help="Threshold for board extraction",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--embedding_layer",
        help="Layer to extract embeddings from",
        type=int,
        default=52,
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip the download step",
    )
    parser.add_argument(
        "--skip_create_table",
        action="store_true",
        help="Skip the table creation step",
    )
    parser.add_argument(
        "--skip_enrich",
        action="store_true",
        help="Skip the enrichment step",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only list files without downloading",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else start_date

    # Parse output folder
    output_folder = Path(args.output_folder) if args.output_folder else None

    # Run pipeline
    results = run_pipeline(
        start_date=start_date,
        end_date=end_date,
        bucket=args.bucket,
        output_folder=output_folder,
        project_name=args.project_name,
        dataset_name=args.dataset_name,
        run_name=args.run_name,
        threshold=args.threshold,
        embedding_layer=args.embedding_layer,
        skip_download=args.skip_download,
        skip_create_table=args.skip_create_table,
        skip_enrich=args.skip_enrich,
        dry_run=args.dry_run,
    )

    # Print results
    if "download_folder" in results:
        print(f"Download folder: {results['download_folder']}")
    if "table" in results:
        print(f"Table URL: {results['table'].url}")
    if "run_url" in results:
        print(f"Run URL: {results['run_url']}")

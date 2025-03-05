#!/usr/bin/env python
"""
Enrich a 3LC table with chess board extraction metrics.

This script processes images in a 3LC table, extracts chess boards using the
BoardExtractor model, and adds metrics like confidence scores and extracted
board images to the table.

Example usage:
    python scripts/process-new-raw/enrich-new-raw-table.py --table_name 2024_11_01_2024_11_01
"""

import argparse
import logging

import cv2
import numpy as np
import tlc
import torch
import torchvision.transforms.v2 as v2

from chessvision.predict.classify_raw import load_board_extractor
from chessvision.predict.extract_board import BoardExtractor
from chessvision.utils import BOARD_SIZE, INPUT_SIZE

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("enrich_tlc_table")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enrich a 3LC table with chess board extraction metrics.",
    )
    parser.add_argument(
        "--table_name",
        help="Name of the 3LC table to process",
        default="2024_11_01_2024_11_01",
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
    return parser.parse_args()


def preprocess_image(sample):
    """
    Preprocess image for model input.

    Args:
        sample: Sample with "image" key

    Returns:
        Preprocessed image tensor
    """
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.ConvertImageDtype(torch.float32),
        ]
    )
    return transforms(sample["image"])


def create_metrics_collector(board_extractor, threshold):
    """
    Create a custom metrics collector function for board extraction.

    Args:
        board_extractor: BoardExtractor instance
        threshold: Threshold for board extraction

    Returns:
        Custom metrics collector function
    """

    def custom_metrics_collector(batch, predictor_output):
        """Extract boards and collect metrics from batch of images."""
        logits = predictor_output.forward

        # Convert logits to probabilities using sigmoid
        probs = torch.sigmoid(logits.squeeze(1))  # Remove channel dimension

        # Get confidence scores (how close to 0 or 1)
        confidence = torch.abs(probs - 0.5) * 2

        # Process each image in batch
        batch_size = confidence.shape[0]
        confidences = []
        extracted_boards = []
        logit_images = []

        for i in range(batch_size):
            # Calculate confidence score
            k = int(confidence[i].numel() * 0.25)
            top_k_confidence = torch.topk(confidence[i].flatten(), k).values
            avg_confidence = float(top_k_confidence.mean())
            confidences.append(avg_confidence)

            # Get original image from batch
            orig_image = (batch[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # Extract board using our BoardExtractor class
            result = board_extractor.extract_board(
                cv2.resize(orig_image, INPUT_SIZE, interpolation=cv2.INTER_AREA),
                orig_image,
                threshold=threshold,
            )

            # Process extracted board
            if result.board_image is not None:
                # Convert to PIL image
                pil_board = Image.fromarray(result.board_image)
                extracted_boards.append(pil_board)
            else:
                # If no board was found, create a black image
                black_image = Image.new("L", BOARD_SIZE, color=0)
                extracted_boards.append(black_image)

            # Process probability mask
            logit_np = result.probability_mask.copy()
            min_val = logit_np.min()
            max_val = logit_np.max()
            if max_val > min_val:  # Avoid division by zero
                logit_np = (logit_np - min_val) / (max_val - min_val)

            # Convert to 8-bit grayscale and enhance contrast
            logit_np = (logit_np * 255).astype(np.uint8)
            logit_image = Image.fromarray(logit_np, mode="L")
            logit_images.append(logit_image)

        return {
            "confidence": confidences,
            "extracted_boards": extracted_boards,
            "logit_images": logit_images,
        }

    return custom_metrics_collector


def main():
    """Enrich a 3LC table with chess board extraction metrics."""
    args = parse_arguments()

    # Generate run name if not provided
    run_name = args.run_name or f"enrich-{args.table_name}"

    logger.info(f"Processing table: {args.table_name}")
    logger.info(f"Project: {args.project_name}, Dataset: {args.dataset_name}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Extraction threshold: {args.threshold}")

    # Load the table
    try:
        table = tlc.Table.from_names(
            args.table_name,
            args.dataset_name,
            args.project_name,
        )
        logger.info(f"Loaded table with {len(table)} rows")
        logger.info(f"Table URL: {table.url}")
    except Exception as e:
        logger.error(f"Error loading table: {e}")
        return

    # Initialize 3LC run
    run = tlc.init(args.project_name, run_name)

    # Load model and create board extractor
    logger.info("Loading board extraction model")
    model = load_board_extractor()
    board_extractor = BoardExtractor(model)

    # Preprocess images
    logger.info("Preprocessing images")
    table.map(preprocess_image)

    # Create metrics collectors
    embedding_collector = tlc.EmbeddingsMetricsCollector([args.embedding_layer])
    custom_collector = create_metrics_collector(board_extractor, args.threshold)

    custom_metrics_collector_schemas = {
        "confidence": tlc.Float("confidence"),
        "extracted_boards": tlc.PILImage("extracted_boards"),
        "logit_images": tlc.PILImage("logit_images"),
    }

    # Collect metrics
    logger.info("Collecting metrics and extracting boards")
    tlc.collect_metrics(
        table,
        [
            tlc.FunctionalMetricsCollector(
                custom_collector,
                column_schemas=custom_metrics_collector_schemas,
                compute_aggregates=False,
            ),
            embedding_collector,
        ],
        tlc.Predictor(
            model,
            layers=[args.embedding_layer],
            preprocess_fn=v2.Resize((256, 256), antialias=True),
        ),
        collect_aggregates=False,
    )

    # Reduce embeddings
    logger.info("Reducing embeddings")
    run.reduce_embeddings_by_foreign_table_url(table.url)

    logger.info(f"Enrichment complete. Run URL: {run.url}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path

import cv2
import numpy as np
import tlc
import torch
import torchvision.transforms.v2 as v2
import tqdm
from PIL import Image, ImageOps

from chessvision.predict.classify_raw import load_board_extractor
from chessvision.predict.extract_board import BoardExtractor
from chessvision.utils import BOARD_SIZE, INPUT_SIZE, get_device

data_dir = Path("../../output/nov-1-2-2024").absolute()
# assert data_dir.exists()

# table_writer = tlc.TableWriter(
#     table_name="raw_data_nov_1_2_2024",
#     dataset_name="chessvision-new-raw",
#     project_name="chessvision-new-raw",
#     column_schemas={
#         "image": tlc.PILImage("image"),
#     },
#     if_exists="overwrite",
# )

# for file in tqdm.tqdm(list(data_dir.iterdir())):
#     if not file.is_file() or file.suffix != ".JPG":
#         assert False

#     pil_img = Image.open(file)
#     table_writer.add_row({"image": pil_img})
#     break

# table = table_writer.finalize()
# print(table.url)

table = tlc.Table.from_names(
    "raw_data_nov_1_2_2025",
    "chessvision-new-raw",
    "chessvision-new-raw",
)

print(table.url)


def map_fn(sample):
    img = sample["image"]
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.ConvertImageDtype(torch.float32),
        ]
    )
    return transforms(img)


table.map(map_fn)

run = tlc.init("chessvision-new-raw", "raw-data-nov-1-2-2024")

model = load_board_extractor()
board_extractor = BoardExtractor(model)


def custom_metrics_collector(batch, predictor_output):
    logits = predictor_output.forward

    # Convert logits to probabilities using sigmoid
    # logits shape is (batch_size, 1, height, width)
    probs = torch.sigmoid(logits.squeeze(1))  # Remove channel dimension

    # Get confidence scores (how close to 0 or 1)
    confidence = torch.abs(probs - 0.5) * 2

    # Process each image in batch
    batch_size = confidence.shape[0]
    confidences = []
    extracted_boards = []
    logit_images = []

    for i in range(batch_size):
        # Take mean of top 25% most confident predictions for each image
        k = int(confidence[i].numel() * 0.25)
        top_k_confidence = torch.topk(confidence[i].flatten(), k).values
        avg_confidence = float(top_k_confidence.mean())
        confidences.append(avg_confidence)

        # Extract board from the image
        prob_np = probs[i].cpu().numpy()

        # Get original image from batch and convert to numpy array in 0-255 range
        orig_image = (batch[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Use our new BoardExtractor class
        result = board_extractor.extract_board(
            cv2.resize(orig_image, INPUT_SIZE, interpolation=cv2.INTER_AREA),
            orig_image,
            threshold=0.3,
        )

        # Add extracted board to results
        if result.board_image is not None:
            # Convert to PIL image
            pil_board = Image.fromarray(result.board_image)
            extracted_boards.append(pil_board)
        else:
            # If no board was found, create a black image
            black_image = Image.new("RGB", (512, 512), color="black")
            extracted_boards.append(black_image)

        # Convert probability mask to grayscale PIL image with enhanced contrast
        # First, ensure full 0-1 range by normalizing
        logit_np = result.probability_mask.copy()
        min_val = logit_np.min()
        max_val = logit_np.max()
        if max_val > min_val:  # Avoid division by zero
            logit_np = (logit_np - min_val) / (max_val - min_val)

        # Convert to 8-bit grayscale
        logit_np = (logit_np * 255).astype(np.uint8)

        # Create PIL image and enhance contrast
        logit_image = Image.fromarray(logit_np, mode="L")
        # logit_image = ImageOps.autocontrast(logit_image, cutoff=0)  # Enhance contrast
        logit_images.append(logit_image)

    return {
        "confidence": confidences,
        "extracted_boards": extracted_boards,
        "logit_images": logit_images,
    }


embedding_collector = tlc.EmbeddingsMetricsCollector([52])
custom_metrics_collector_schemas = {
    "confidence": tlc.Float("confidence"),
    "extracted_boards": tlc.PILImage("extracted_boards"),
    "logit_images": tlc.PILImage("logit_images"),
}
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
        layers=[52],
        preprocess_fn=v2.Resize((256, 256), antialias=True),
    ),
    collect_aggregates=False,
)

run.reduce_embeddings_by_foreign_table_url(table.url)

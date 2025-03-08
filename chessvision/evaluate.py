from __future__ import annotations

import argparse
import io
import logging
import time
from collections.abc import Generator
from pathlib import Path

import cairosvg
import chess
import chess.svg
import cv2
import numpy as np
import tlc
from PIL import Image
from tqdm import tqdm

from chessvision.core import ChessVision

logger = logging.getLogger(__name__)

TEST_DATA_DIR = Path(ChessVision.DATA_ROOT) / "test"
PROJECT_NAME = "chessvision-testing"
LABEL_NAMES = ChessVision.LABEL_NAMES


def accuracy(a: list[str], b: list[str]) -> float:
    return sum([aa == bb for aa, bb in zip(a, b)]) / len(a)


def top_k_accuracy(predictions: np.ndarray, true_labels: np.ndarray, k: int = 3) -> tuple[float, ...]:
    """
    predictions: (64, 13) probability distributions
    truth      : (64, 1)  true labels
    k          : compute top-1, top-2, top-3 accuracies

    returns    : tuple containing accuracies for k=1, k=2, k=3
    """
    sorted_predictions = np.argsort(predictions, axis=1)
    top_k = sorted_predictions[:, -k:]
    top_k_predictions = np.array([["" for _ in range(k)] for _ in range(64)])
    hits = [0] * k

    for i in range(64):
        for j in range(k):
            try:
                _top_k = list(ChessVision.LABEL_INDICES.keys())[top_k[i, j]]
            except IndexError:
                _top_k = "f"
            top_k_predictions[i, j] = _top_k

    for square_ind in range(64):
        for j in range(k):
            if true_labels[square_ind] in top_k_predictions[square_ind, -j - 1 :]:
                hits[j] += 1

    accuracies = [hit / 64 for hit in hits]
    return tuple(accuracies)


def snake(squares: list[str]) -> list[str]:
    assert len(squares) == 64
    for i in range(8):
        squares[i * 8 : (i + 1) * 8] = list(reversed(squares[i * 8 : (i + 1) * 8]))
    return list(reversed(squares))


def vectorize_chessboard(board: chess.Board) -> list[str]:
    res = list("f" * 64)

    piecemap = board.piece_map()

    for i in range(64):
        piece = piecemap.get(i)
        if piece:
            res[i] = piece.symbol()

    return res


def get_test_generator(image_dir: Path) -> Generator[tuple[str, np.ndarray], None, None]:
    img_filenames = ChessVision._listdir_nohidden(str(image_dir))
    test_imgs = np.array([cv2.imread(str(image_dir / x)) for x in img_filenames])

    for i in range(len(test_imgs)):
        yield img_filenames[i], test_imgs[i]


def chessboard_to_pil_image(chessboard: chess.Board) -> Image.Image:
    svg = chess.svg.board(chessboard)
    buffer = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
    return Image.open(io.BytesIO(buffer))


def save_svg(chessboard: chess.Board, path: Path) -> None:
    svg = chess.svg.board(chessboard)
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(path))


def evaluate_model(
    image_folder: Path = TEST_DATA_DIR / "raw",
    truth_folder: Path = TEST_DATA_DIR / "ground_truth",
    run: tlc.Run | None = None,
    threshold: float = 0.5,
    project_name: str = PROJECT_NAME,
    board_extractor_weights: str | None = None,
    classifier_weights: str | None = None,
) -> tlc.Run:
    """Run evaluation on test images using the ChessVision model.

    This script evaluates the ChessVision model on a test dataset, computing various
    metrics including top-k accuracy, extraction success rate, and processing time.
    Results are logged to a 3LC run for visualization and analysis.

    Args:
        image_folder: Directory containing test images
        truth_folder: Directory containing ground truth labels
        run: Optional 3LC run to log results to
        threshold: Confidence threshold for board extraction
        project_name: Name of the 3LC project
        board_extractor_weights: Optional path to board extractor weights
        classifier_weights: Optional path to classifier weights

    Returns:
        The 3LC run containing evaluation results
    """
    if not run:
        run = tlc.init(project_name)

    # Initialize ChessVision model with optional weights
    cv = ChessVision(
        board_extractor_weights=board_extractor_weights,
        classifier_weights=classifier_weights,
    )

    # Set up metrics writer
    metrics_writer = tlc.MetricsTableWriter(
        run.url,
        column_schemas={
            "raw_img": tlc.Schema(value=tlc.ImageUrlStringValue("raw_img")),
            "true_labels": tlc.CategoricalLabel("true_labels", LABEL_NAMES),
            "predicted_labels": tlc.CategoricalLabel("predicted_labels", LABEL_NAMES),
            "rendered_board": tlc.Schema(value=tlc.ImageUrlStringValue("rendered_board")),
            "extracted_board": tlc.Schema(value=tlc.ImageUrlStringValue("extracted_board")),
            "predicted_masks": tlc.Schema(value=tlc.ImageUrlStringValue("predicted_masks")),
        },
    )

    data_generator = get_test_generator(image_folder)

    top_1_accuracy = 0.0
    top_2_accuracy = 0.0
    top_3_accuracy = 0.0

    times = []
    test_set_size = len(ChessVision._listdir_nohidden(str(image_folder)))
    extraction_failures = 0

    with tlc.bulk_data_url_context(run.bulk_data_url, metrics_writer.url):
        for index, (filename, img) in tqdm(enumerate(data_generator), total=test_set_size, desc="Evaluating images"):
            result = cv.process_image(img)
            times.append(result.processing_time)

            # Save the predicted mask
            predicted_mask_url = Path((run.bulk_data_url / "predicted_masks" / (filename[:-4] + ".png")).to_str())
            predicted_mask_url.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result.board_extraction.binary_mask).save(predicted_mask_url)

            if result.position is None:
                extraction_failures += 1
                logger.warning(f"Failed to extract board from {filename}")

                metrics_batch = {
                    "raw_img": [str(image_folder / filename)],
                    "predicted_masks": [str(predicted_mask_url)],
                    "extracted_board": [str(ChessVision.BLACK_BOARD_PATH)],
                    "rendered_board": [""],
                    "example_id": [index],
                    "is_failed": [True],
                    "accuracy": [0.0],
                    "square_crop": [ChessVision.BLACK_SQUARE_PATH],
                    "true_labels": [0],
                    "predicted_labels": [0],
                }

                metrics_writer.add_batch(metrics_batch)
                continue

            assert result.board_extraction.board_image is not None

            # Save the extracted board
            extracted_board_url = Path((run.bulk_data_url / "extracted_board" / (filename[:-4] + ".png")).to_str())
            extracted_board_url.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result.board_extraction.board_image).save(extracted_board_url)

            # Load the true labels
            truth_file = truth_folder / (filename[:-4] + ".txt")

            with truth_file.open("r") as truth:
                true_labels = truth.read()

            true_labels = snake(list(true_labels))
            true_labels_int = [LABEL_NAMES.index(label) for label in true_labels]

            # Compute the accuracy
            predictions = result.position.predictions
            top_1, top_2, top_3 = top_k_accuracy(predictions, true_labels, k=3)
            top_1_accuracy += top_1
            top_2_accuracy += top_2
            top_3_accuracy += top_3

            # Register the predicted labels
            board = chess.Board(result.position.fen)
            predicted_labels = vectorize_chessboard(board)
            predicted_labels = snake(predicted_labels)
            predicted_labels_int = [LABEL_NAMES.index(label) for label in predicted_labels]

            # Write and register a rendered board image
            svg_url = Path((run.bulk_data_url / "rendered_board" / (filename[:-4] + ".png")).to_str())
            svg_url.parent.mkdir(parents=True, exist_ok=True)
            save_svg(board, svg_url)

            metrics_batch = {
                "raw_img": [str(image_folder / filename)] * 64,
                "predicted_masks": [str(predicted_mask_url)] * 64,
                "extracted_board": [str(extracted_board_url)] * 64,
                "rendered_board": [str(svg_url)] * 64,
                "accuracy": [top_1] * 64,
                "square_crop": [Image.fromarray(img.squeeze()) for img in result.position.squares],
                "true_labels": true_labels_int,
                "predicted_labels": predicted_labels_int,
                "example_id": [index] * 64,
                "is_failed": [False] * 64,
            }

            metrics_writer.add_batch(metrics_batch)

    top_1_accuracy /= test_set_size
    top_2_accuracy /= test_set_size
    top_3_accuracy /= test_set_size

    aggregate_data = {
        "top_1_accuracy": f"{top_1_accuracy: .3f}",
        "top_2_accuracy": f"{top_2_accuracy: .3f}",
        "top_3_accuracy": f"{top_3_accuracy: .3f}",
        "avg_time_per_prediction": sum(times) / test_set_size,
        "extraction_failures": extraction_failures,
        "board_extractor_weights": cv._board_extractor_weights,
        "classifier_weights": cv._classifier_weights,
    }

    run.set_parameters(
        {
            "test_results": aggregate_data,
            "threshold": threshold,
        },
    )

    logger.info(f"Evaluated {test_set_size} raw images")
    metrics_table = metrics_writer.finalize()
    run.add_metrics_table(metrics_table)
    run.set_status_completed()
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ChessVision model on test dataset")
    parser.add_argument("--image-folder", type=str, default=str(TEST_DATA_DIR / "raw"))
    parser.add_argument("--truth-folder", type=str, default=str(TEST_DATA_DIR / "ground_truth"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--project-name", type=str, default="chessvision-testing")
    parser.add_argument("--board-extractor-weights", type=str, help="Path to board extractor weights")
    parser.add_argument("--classifier-weights", type=str, help="Path to classifier weights")

    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Running ChessVision evaluation...")
    args = parse_args()
    start = time.time()

    run = evaluate_model(
        image_folder=Path(args.image_folder),
        truth_folder=Path(args.truth_folder),
        project_name=args.project_name,
        threshold=args.threshold,
        board_extractor_weights=args.board_extractor_weights,
        classifier_weights=args.classifier_weights,
    )
    stop = time.time()
    logger.info(f"Evaluation completed in {stop - start:.1f}s")
    if "test_results" in run.constants["parameters"]:
        logger.info("Test accuracy: {}".format(run.constants["parameters"]["test_results"]["top_1_accuracy"]))
        logger.info("Top-2 accuracy: {}".format(run.constants["parameters"]["test_results"]["top_2_accuracy"]))
        logger.info("Top-3 accuracy: {}".format(run.constants["parameters"]["test_results"]["top_3_accuracy"]))

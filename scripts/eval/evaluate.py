from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cairosvg
import chess
import chess.svg
import cv2
import numpy as np
import tlc
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from chessvision import ChessVision, constants, utils

logger = logging.getLogger(__name__)

TEST_DATA_DIR = constants.DATA_ROOT / "test"


@dataclass
class TopKAccuracyResult:
    """Results from top-k accuracy computation."""

    k: int
    accuracies: Sequence[float]

    @property
    def top_1(self) -> float:
        """Get top-1 accuracy."""
        return self.accuracies[0]

    @property
    def top_2(self) -> float:
        """Get top-2 accuracy if available."""
        return self.accuracies[1] if len(self.accuracies) > 1 else 0.0

    @property
    def top_3(self) -> float:
        """Get top-3 accuracy if available."""
        return self.accuracies[2] if len(self.accuracies) > 2 else 0.0


@dataclass
class PositionMetrics:
    """Metrics for a single position evaluation."""

    top_k_accuracy: TopKAccuracyResult
    true_labels: list[str]
    predicted_labels: list[str]
    true_labels_indices: list[int]
    predicted_labels_indices: list[int]


def board_to_labels(board: chess.Board) -> list[str]:
    """Convert chess.Board to list of piece labels.

    Args:
        board: Chess board

    Returns:
        List of 64 piece symbols in FEN order (a8-h8, a7-h7, ..., a1-h1)
    """
    labels = ["f"] * 64  # Empty squares

    # Fill in pieces in chess.Board order (a1-h1, a2-h2, ..., a8-h8)
    board_labels = ["f"] * 64
    for square, piece in board.piece_map().items():
        board_labels[square] = piece.symbol()

    # Convert to FEN order (a8-h8, a7-h7, ..., a1-h1)
    for rank in range(8):
        for file in range(8):
            # Convert from board index to FEN index
            board_idx = rank * 8 + file  # 0-63 from a1
            fen_idx = (7 - rank) * 8 + file  # 0-63 from a8
            labels[fen_idx] = board_labels[board_idx]

    return labels


def compute_top_k_accuracy(
    predictions: NDArray[np.float32],
    true_labels: list[str],
    k: int = 3,
) -> TopKAccuracyResult:
    """Compute top-k accuracy for piece classification.

    Args:
        predictions: (64, 13) probability distributions
        true_labels: List of 64 true label strings
        k: Number of top predictions to consider

    Returns:
        TopKAccuracyResult containing accuracies for k=1,2,3
    """
    sorted_predictions = np.argsort(predictions, axis=1)
    top_k_indices = sorted_predictions[:, -k:]
    hits = [0] * k

    for square_idx in range(64):
        true_label = true_labels[square_idx]
        # Check each prediction against ground truth
        for k_idx in range(k):
            pred_idx = top_k_indices[square_idx, -(k_idx + 1)]
            pred_label = constants.LABEL_NAMES[pred_idx]
            if pred_label == true_label:
                # If correct at k_idx, it's correct for all higher k values
                for j in range(k_idx, k):
                    hits[j] += 1
                break

    accuracies = [hit / 64 for hit in hits]
    return TopKAccuracyResult(k=k, accuracies=accuracies)


def compute_position_metrics(
    predictions: NDArray[np.float32],
    true_fen: str,
    k: int = 3,
) -> PositionMetrics:
    """Compute metrics for a single position.

    Args:
        predictions: Model predictions (64, 13)
        true_fen: Ground truth position as FEN string
        k: Number of top predictions to consider

    Returns:
        PositionMetrics containing accuracy and label information
    """
    true_board = chess.Board(true_fen + " w KQkq - 0 1")
    true_labels = board_to_labels(true_board)
    true_labels_indices = [constants.LABEL_INDICES[label] for label in true_labels]

    # Get predicted labels
    predicted_labels_indices = np.argmax(predictions, axis=1).tolist()
    predicted_labels = [constants.LABEL_NAMES[idx] for idx in predicted_labels_indices]

    # Compute accuracies
    top_k = compute_top_k_accuracy(predictions, true_labels, k=k)

    return PositionMetrics(
        top_k_accuracy=top_k,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        true_labels_indices=true_labels_indices,
        predicted_labels_indices=predicted_labels_indices,
    )


def get_test_generator(test_table: tlc.Table) -> Generator[tuple[np.ndarray, str, str], None, None]:
    """Returns (img, filename, true_fen)"""
    for img in test_table:
        img_url: str = img._tlc_url
        img_array = cv2.imread(img_url)
        filename = img_url.split("/")[-1]
        fen_path = img_url.lower().replace("raw", "ground_truth").replace("jpg", "txt")
        with Path(fen_path).open("r") as f:
            true_fen = f.read().strip()
        yield img_array, filename, true_fen


def save_svg(chessboard: chess.Board, path: Path) -> None:
    svg = chess.svg.board(chessboard, size=512)
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(path))


def resolve_table(
    *,
    table_name: str,
    project_name: str = "chessvision-testing",
    dataset_name: str = "test",
    image_folder: Path | None = None,
) -> tlc.Table:
    """Resolve table by first trying to load existing, then creating if needed.

    Args:
        table_name: Name of the table to resolve
        project_name: Name of the project
        dataset_name: Name of the dataset
        image_folder: Path to folder containing images

    Returns:
        Resolved tlc.Table instance
    """
    try:
        table = tlc.Table.from_names(
            table_name=table_name,
            dataset_name=dataset_name,
            project_name=project_name,
        )
        logger.info(f"Resolved existing table: {table_name} ({len(table)} images)")
    except FileNotFoundError as err:
        if image_folder is None:
            raise ValueError("image_folder is required if table does not exist") from err  # noqa: TRY003
        table = tlc.Table.from_image_folder(
            image_folder,
            include_label_column=False,
            extensions=(".JPG", ".jpg"),
            dataset_name=dataset_name,
            table_name=table_name,
            project_name=project_name,
            add_weight_column=False,
            if_exists="reuse",
        )
        logger.info(f"Created new table: {table.name} ({len(table)} images)")
    return table


def evaluate_model(
    *,
    image_folder: Path | None = None,
    run: tlc.Run | None = None,
    threshold: float = 0.5,
    project_name: str = "chessvision-testing",
    table_name: str = "initial",
    run_name: str = "",
    run_description: str = "",
    board_extractor_weights: str | None = None,
    classifier_weights: str | None = None,
    classifier_model_id: str = "",
) -> tlc.Run:
    """Run evaluation on test images using the ChessVision model.

    This script evaluates the ChessVision model on a test dataset, computing various
    metrics including top-k accuracy, extraction success rate, and processing time.
    Results are logged to a 3LC run for visualization and analysis.

    Args:
        image_folder: Directory containing test images
        run: Optional 3LC run to log results to
        threshold: Confidence threshold for board extraction
        project_name: Name of the 3LC project
        board_extractor_weights: Optional path to board extractor weights
        classifier_weights: Optional path to classifier weights

    Returns:
        The 3LC run containing evaluation results
    """
    # Get or create tlc.Table using the helper function
    test_table = resolve_table(
        table_name=table_name,
        image_folder=image_folder,
        project_name=project_name,
    )

    if not run:
        run = tlc.init(
            project_name=project_name,
            run_name=run_name,
            description=run_description,
        )

    # Initialize ChessVision model with optional weights
    cv = ChessVision(
        board_extractor_weights=board_extractor_weights,
        classifier_weights=classifier_weights,
        classifier_model_id=classifier_model_id,
        lazy_load=False,
    )

    # Set up metrics writer
    metrics_writer = tlc.MetricsTableWriter(
        run_url=run.url,
        foreign_table_url=test_table.url,
        column_schemas={
            "true_labels": tlc.CategoricalLabel("true_labels", constants.LABEL_NAMES),
            "predicted_labels": tlc.CategoricalLabel("predicted_labels", constants.LABEL_NAMES),
            "rendered_board": tlc.Schema(value=tlc.ImageUrlStringValue("rendered_board")),
            "extracted_board": tlc.Schema(value=tlc.ImageUrlStringValue("extracted_board")),
            "predicted_masks": tlc.Schema(value=tlc.ImageUrlStringValue("predicted_masks")),
        },
    )

    data_generator = get_test_generator(test_table)

    top_1_accuracy = 0.0
    top_2_accuracy = 0.0
    top_3_accuracy = 0.0

    times = []
    test_set_size = len(test_table)
    extraction_failures = 0

    with tlc.bulk_data_url_context(run.bulk_data_url, metrics_writer.url):
        for index, (img, filename, true_fen) in tqdm(
            enumerate(data_generator),
            total=test_set_size,
            desc="Evaluating images",
        ):
            result = cv.process_image(img)
            times.append(result.processing_time)

            # Save the predicted mask
            predicted_mask_url = Path((run.bulk_data_url / "predicted_masks" / (filename[:-4] + ".png")).to_str())
            predicted_mask_url.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result.board_extraction.binary_mask).save(predicted_mask_url)

            if result.position is None:
                extraction_failures += 1

                metrics_batch = {
                    "predicted_masks": [str(predicted_mask_url)],
                    "extracted_board": [str(constants.BLACK_BOARD_PATH)],
                    "rendered_board": [""],
                    "example_id": [index],
                    "is_failed": [True],
                    "accuracy": [0.0],
                    "square_crop": [Image.open(constants.BLACK_SQUARE_PATH)],
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

            # Save the rendered board image
            board = chess.Board(result.position.fen)
            svg_url = Path((run.bulk_data_url / "rendered_board" / (filename[:-4] + ".png")).to_str())
            svg_url.parent.mkdir(parents=True, exist_ok=True)
            save_svg(board, svg_url)

            # Compute the accuracy
            predictions = result.position.predictions
            metrics = compute_position_metrics(predictions, true_fen)

            top_1 = metrics.top_k_accuracy.top_1
            top_2 = metrics.top_k_accuracy.top_2
            top_3 = metrics.top_k_accuracy.top_3

            top_1_accuracy += top_1
            top_2_accuracy += top_2
            top_3_accuracy += top_3

            metrics_batch = {
                "predicted_masks": [str(predicted_mask_url)] * 64,
                "extracted_board": [str(extracted_board_url)] * 64,
                "rendered_board": [str(svg_url)] * 64,
                "accuracy": [top_1] * 64,
                "square_crop": [Image.fromarray(img.squeeze()) for img in result.position.squares],
                "true_labels": metrics.true_labels_indices,
                "predicted_labels": metrics.predicted_labels_indices,
                "example_id": [index] * 64,
                "is_failed": [False] * 64,
            }

            metrics_writer.add_batch(metrics_batch)

    top_1_accuracy /= test_set_size - extraction_failures
    top_2_accuracy /= test_set_size - extraction_failures
    top_3_accuracy /= test_set_size - extraction_failures

    aggregate_data = {
        "top_1_accuracy": f"{top_1_accuracy: .3f}",
        "top_2_accuracy": f"{top_2_accuracy: .3f}",
        "top_3_accuracy": f"{top_3_accuracy: .3f}",
        "avg_time_per_prediction": sum(times) / test_set_size,
        "extraction_failures": extraction_failures,
        "board_extractor_weights": cv._board_extractor_weights,
        "classifier_weights": cv._classifier_weights,
        "test_table_name": table_name,
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
    parser.add_argument("--image-folder", type=str, default=str(TEST_DATA_DIR / "initial" / "raw"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--project-name", type=str, default="chessvision-testing")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--run-description", type=str, default="")
    parser.add_argument("--board-extractor-weights", type=str, help="Path to board extractor weights")
    parser.add_argument("--classifier-weights", type=str, help="Path to classifier weights")
    parser.add_argument("--classifier-model-id", type=str, default="yolo", help="Classifier model ID")
    parser.add_argument("--table-name", type=str, default="initial", help="Table name")
    return parser.parse_args()


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


if __name__ == "__main__":
    logger = setup_logger(__name__)

    logger.info("Running ChessVision evaluation...")
    args = parse_args()
    logger.info(f"Arguments: {args}")

    start = time.time()

    run = evaluate_model(
        image_folder=Path(args.image_folder),
        run=None,
        project_name=args.project_name,
        run_name=args.run_name,
        run_description=args.run_description,
        threshold=args.threshold,
        board_extractor_weights=args.board_extractor_weights,
        classifier_weights=args.classifier_weights,
        classifier_model_id=args.classifier_model_id,
        table_name=args.table_name,
    )
    stop = time.time()
    logger.info(f"Evaluation completed in {stop - start:.1f}s")
    if "test_results" in run.constants["parameters"]:
        logger.info("Test accuracy: {}".format(run.constants["parameters"]["test_results"]["top_1_accuracy"]))
        logger.info("Top-2 accuracy: {}".format(run.constants["parameters"]["test_results"]["top_2_accuracy"]))
        logger.info("Top-3 accuracy: {}".format(run.constants["parameters"]["test_results"]["top_3_accuracy"]))
        logger.info(
            "Extraction failures: {}".format(run.constants["parameters"]["test_results"]["extraction_failures"]),
        )

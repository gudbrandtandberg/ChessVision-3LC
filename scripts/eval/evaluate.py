from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Generator, Sequence
from contextlib import nullcontext
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

from chessvision import ChessVision, constants
from chessvision.cv_types import PositionResult

logger = logging.getLogger(__name__)

TEST_DATA_DIR = constants.DATA_ROOT / "test"


@dataclass
class PositionAccuracy:
    """Simple accuracy metrics for a position."""

    accuracy: float  # Percentage of correctly classified squares
    num_correct: int  # Number of correctly classified squares
    total_squares: int = 64  # Total number of squares (always 64 for chess)


def compute_position_accuracy(predicted_fen: str, true_fen: str) -> PositionAccuracy:
    """Compute accuracy between two FEN strings."""
    pred_board = chess.BaseBoard(predicted_fen)
    true_board = chess.BaseBoard(true_fen)

    correct = 0
    for square in chess.SQUARES:
        pred_piece = pred_board.piece_at(square)
        true_piece = true_board.piece_at(square)
        if pred_piece == true_piece:  # This handles None (empty) squares correctly
            correct += 1

    return PositionAccuracy(
        accuracy=correct / 64,
        num_correct=correct,
    )


def evaluate_position(result: PositionResult, true_fen: str) -> tuple[PositionAccuracy, PositionAccuracy]:
    """Evaluate both original and validated position accuracy."""
    original_accuracy = compute_position_accuracy(result.original_fen, true_fen)
    validated_accuracy = compute_position_accuracy(result.fen, true_fen)
    return original_accuracy, validated_accuracy


def board_to_labels(board: chess.BaseBoard) -> list[str]:
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


def compute_model_topk_accuracy(
    model_probabilities: NDArray[np.float32],
    true_fen: str,
    k: int = 3,
) -> TopKAccuracyResult:
    """Compute top-k accuracy for the raw model predictions."""
    # Get true labels from FEN
    board = chess.BaseBoard(true_fen)
    true_labels = board_to_labels(board)

    # Get top-k predictions
    sorted_predictions = np.argsort(model_probabilities, axis=1)
    top_k_indices = sorted_predictions[:, -k:]  # This gets the highest k indices
    hits = [0] * k

    for square_idx in range(64):
        true_label = true_labels[square_idx]
        # Check each prediction against ground truth
        for k_idx in range(k):
            pred_idx = top_k_indices[square_idx, -(k_idx + 1)]  # Going from highest to lowest
            pred_label = constants.LABEL_NAMES[pred_idx]
            if pred_label == true_label:
                # If correct at k_idx, it's correct for all higher k values
                for j in range(k_idx, k):
                    hits[j] += 1
                break

    accuracies = [hit / 64 for hit in hits]
    return TopKAccuracyResult(k=k, accuracies=accuracies)


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
    svg = chess.svg.board(chessboard)
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
    board_extractor_model_id: str = "",
    classifier_weights: str | None = None,
    classifier_model_id: str = "",
    include_metrics_table: bool = False,
) -> tlc.Run:
    """Run evaluation on test images using the ChessVision model."""
    # Initialize run and model
    test_table = resolve_table(
        table_name=table_name,
        image_folder=image_folder,
        project_name=project_name,
    )
    if not run:
        run = tlc.init(project_name=project_name, run_name=run_name, description=run_description)

    cv = ChessVision(
        board_extractor_weights=board_extractor_weights,
        board_extractor_model_id=board_extractor_model_id,
        classifier_weights=classifier_weights,
        classifier_model_id=classifier_model_id,
        lazy_load=False,
    )

    # Initialize counters
    total_original_accuracy = 0.0
    total_validated_accuracy = 0.0
    total_top2_accuracy = 0.0
    total_top3_accuracy = 0.0
    validation_improvements = 0
    validation_fixes = 0
    extraction_failures = 0
    times = []
    test_set_size = len(test_table)

    # Setup metrics writer only if needed
    metrics_writer = None
    if include_metrics_table:
        metrics_writer = tlc.MetricsTableWriter(
            run_url=run.url,
            foreign_table_url=test_table.url,
            column_schemas={
                "true_labels": tlc.Schema(
                    value=tlc.Int32Value(
                        value_min=0,
                        value_max=12,
                        number_role=tlc.NUMBER_ROLE_LABEL,
                        value_map=tlc.MapElement._construct_value_map(constants.LABEL_NAMES),
                    ),
                    writable=True,
                ),
                "predicted_labels": tlc.Schema(
                    value=tlc.Int32Value(
                        value_min=0,
                        value_max=12,
                        number_role=tlc.NUMBER_ROLE_LABEL,
                        value_map=tlc.MapElement._construct_value_map(constants.LABEL_NAMES),
                    ),
                ),
                "validated_labels": tlc.CategoricalLabel("validated_labels", constants.LABEL_NAMES),
                "rendered_board_original": tlc.Schema(value=tlc.ImageUrlStringValue("rendered_board_original")),
                "rendered_board_validated": tlc.Schema(value=tlc.ImageUrlStringValue("rendered_board_validated")),
                "extracted_board": tlc.Schema(value=tlc.ImageUrlStringValue("extracted_board")),
                "predicted_masks": tlc.Schema(value=tlc.ImageUrlStringValue("predicted_masks")),
            },
        )

    # Main evaluation loop
    context_manager = (
        tlc.bulk_data_url_context(run.bulk_data_url, metrics_writer.url) if metrics_writer else nullcontext()
    )

    with context_manager:
        for index, (img, filename, true_fen) in tqdm(
            enumerate(get_test_generator(test_table)),
            total=test_set_size,
            desc="Evaluating images",
        ):
            # Process image
            result = cv.process_image(img)
            times.append(result.processing_time)

            # Handle extraction failure
            if result.position is None:
                extraction_failures += 1
                if metrics_writer:
                    mask_url = save_predicted_mask(run, filename, result.board_extraction.binary_mask)

                    metrics_batch = {
                        "predicted_masks": [str(mask_url)],
                        "extracted_board": [str(constants.BLACK_BOARD_PATH)],
                        "rendered_board_original": [""],
                        "rendered_board_validated": [""],
                        "example_id": [index],
                        "is_failed": [True],
                        "top_1_accuracy": [0.0],
                        "top_1_accuracy_validated": [0.0],
                        "top_2_accuracy": [0.0],
                        "top_3_accuracy": [0.0],
                        "num_fixes": [0],
                        "square_crop": [Image.open(constants.BLACK_SQUARE_PATH)],
                        "true_labels": [0],
                        "predicted_labels": [0],
                        "validated_labels": [0],
                    }
                    metrics_writer.add_batch(metrics_batch)
                continue

            # Process successful extraction
            assert result.board_extraction.board_image is not None

            # Compute all metrics
            original_accuracy = compute_position_accuracy(result.position.original_fen, true_fen)
            validated_accuracy = compute_position_accuracy(result.position.fen, true_fen)
            topk_acc = compute_model_topk_accuracy(result.position.model_probabilities, true_fen)
            pred_indices, true_indices = get_label_indices(result.position.model_probabilities, true_fen)
            validated_indices = get_validated_indices(result.position.fen)

            # Update running totals
            total_original_accuracy += original_accuracy.accuracy
            total_validated_accuracy += validated_accuracy.accuracy
            total_top2_accuracy += topk_acc.top_2
            total_top3_accuracy += topk_acc.top_3
            if validated_accuracy.accuracy > original_accuracy.accuracy:
                validation_improvements += 1
            validation_fixes += len(result.position.validation_fixes)

            # Add detailed metrics only if requested
            if metrics_writer:
                mask_url = save_predicted_mask(run, filename, result.board_extraction.binary_mask)
                board_url = save_extracted_board(run, filename, result.board_extraction.board_image)
                original_svg_url = save_predicted_board(run, filename, result.position.original_fen, suffix="original")
                validated_svg_url = save_predicted_board(run, filename, result.position.fen, suffix="validated")
                pred_indices, true_indices = get_label_indices(result.position.model_probabilities, true_fen)
                validated_indices = get_validated_indices(result.position.fen)

                metrics_batch = {
                    "predicted_masks": [str(mask_url)] * 64,
                    "extracted_board": [str(board_url)] * 64,
                    "rendered_board_original": [str(original_svg_url)] * 64,
                    "rendered_board_validated": [str(validated_svg_url)] * 64,
                    "top_1_accuracy_validated": [validated_accuracy.accuracy] * 64,
                    "top_1_accuracy": [original_accuracy.accuracy] * 64,
                    "top_2_accuracy": [topk_acc.top_2] * 64,
                    "top_3_accuracy": [topk_acc.top_3] * 64,
                    "num_fixes": [len(result.position.validation_fixes)] * 64,
                    "square_crop": [Image.fromarray(img.squeeze()) for img in result.position.squares],
                    "example_id": [index] * 64,
                    "is_failed": [False] * 64,
                    "true_labels": true_indices,
                    "predicted_labels": pred_indices,
                    "validated_labels": validated_indices,
                }
                metrics_writer.add_batch(metrics_batch)

    # Compute final metrics
    successful_evaluations = test_set_size - extraction_failures
    aggregate_data = {
        "top_1_accuracy_validated": total_validated_accuracy / successful_evaluations,
        "top_1_accuracy": total_original_accuracy / successful_evaluations,
        "top_2_accuracy": total_top2_accuracy / successful_evaluations,
        "top_3_accuracy": total_top3_accuracy / successful_evaluations,
        "validation_fixes": validation_fixes,
        "validation_improvements": validation_improvements,
        "extraction_failures": extraction_failures,
        "avg_time_per_prediction": sum(times) / test_set_size,
        "board_extractor_weights": cv._board_extractor_weights,
        "classifier_weights": cv._classifier_weights,
        "test_table_name": table_name,
    }

    # Finalize run
    run.set_parameters({"test_results": aggregate_data, "threshold": threshold})

    # Only add metrics table if requested
    if metrics_writer:
        metrics_table = metrics_writer.finalize()
        run.add_metrics_table(metrics_table)

    run.set_status_completed()
    return run


def save_predicted_board(run: tlc.Run, filename: str, fen: str, suffix: str = "") -> Path:
    """Save rendered board as PNG.

    Args:
        run: Current evaluation run
        filename: Base filename
        fen: FEN string to render
        suffix: Optional suffix to add to filename (e.g., "original" or "validated")
    """
    board = chess.Board(fen)
    name_stem = filename[:-4]  # Remove .jpg
    if suffix:
        name_stem = f"{name_stem}_{suffix}"
    svg_url = Path((run.bulk_data_url / "rendered_board" / (name_stem + ".png")).to_str())
    svg_url.parent.mkdir(parents=True, exist_ok=True)
    save_svg(board, svg_url)
    return svg_url


def save_extracted_board(run: tlc.Run, filename: str, board_image: NDArray[np.uint8]) -> Path:
    extracted_board_url = Path((run.bulk_data_url / "extracted_board" / (filename[:-4] + ".png")).to_str())
    extracted_board_url.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(board_image).save(extracted_board_url)
    return extracted_board_url


def save_predicted_mask(run: tlc.Run, filename: str, binary_mask: NDArray[np.uint8]) -> Path:
    predicted_mask_url = Path((run.bulk_data_url / "predicted_masks" / (filename[:-4] + ".png")).to_str())
    predicted_mask_url.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary_mask).save(predicted_mask_url)
    return predicted_mask_url


def get_label_indices(
    probabilities: NDArray[np.float32],
    true_fen: str,
) -> tuple[list[int], list[int]]:
    """Get predicted and true label indices.

    Args:
        probabilities: Model probability outputs (64, 13)
        true_fen: Ground truth FEN string

    Returns:
        Tuple of (predicted_indices, true_indices if true_fen provided else None)
    """
    # Get predicted indices from probabilities
    pred_indices = np.argmax(probabilities, axis=1).tolist()

    # Get true indices
    board = chess.BaseBoard(true_fen)
    true_labels = board_to_labels(board)
    true_indices = [constants.LABEL_NAMES.index(label) for label in true_labels]
    return pred_indices, true_indices


def get_validated_indices(fen: str) -> list[int]:
    """Get label indices from a FEN string.

    Args:
        fen: FEN string to convert to indices

    Returns:
        List of 64 label indices corresponding to the position
    """
    board = chess.BaseBoard(fen)
    labels = board_to_labels(board)
    return [constants.LABEL_NAMES.index(label) for label in labels]


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
    parser.add_argument("--board-extractor-model-id", type=str, default=None, help="Board extractor model ID")
    parser.add_argument("--table-name", type=str, default="initial", help="Table name")
    parser.add_argument("--include-metrics-table", action="store_true", help="Include metrics table")
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
        board_extractor_model_id=args.board_extractor_model_id,
        classifier_weights=args.classifier_weights,
        classifier_model_id=args.classifier_model_id,
        table_name=args.table_name,
        include_metrics_table=args.include_metrics_table,
    )
    stop = time.time()
    logger.info(f"Evaluation completed in {stop - start:.1f}s")
    if "test_results" in run.constants["parameters"]:
        logger.info("Test accuracy: {:.3f}".format(run.constants["parameters"]["test_results"]["top_1_accuracy"]))
        logger.info(
            "Validated accuracy: {:.3f}".format(
                run.constants["parameters"]["test_results"]["top_1_accuracy_validated"],
            ),
        )
        logger.info(
            "Validation improvements: {}".format(
                run.constants["parameters"]["test_results"]["validation_improvements"],
            ),
        )
        logger.info(
            "Validation fixes: {}".format(
                run.constants["parameters"]["test_results"]["validation_fixes"],
            ),
        )
        logger.info(
            "Extraction failures: {}".format(run.constants["parameters"]["test_results"]["extraction_failures"]),
        )

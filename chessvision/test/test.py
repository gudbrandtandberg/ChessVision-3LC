import argparse
import time
from pathlib import Path

import cairosvg
import chess
import chess.svg
import cv2
import numpy as np
import tlc
import torch
from PIL import Image
from tqdm import tqdm

from chessvision.predict.classify_raw import classify_raw
from chessvision.utils import DATA_ROOT, INPUT_SIZE, PIECE_SIZE, labels, listdir_nohidden

TEST_DATA_DIR = Path(DATA_ROOT) / "test"

PROJECT_NAME = "chessvision-testing"
LABEL_NAMES = ["f", "P", "p", "R", "r", "N", "n", "B", "b", "Q", "q", "K", "k"]

BLACK_BOARD = Image.fromarray(np.zeros(INPUT_SIZE).astype(np.uint8))
BLACK_CROP = Image.fromarray(np.zeros(PIECE_SIZE).astype(np.uint8))


def accuracy(a, b):
    return sum([aa == bb for aa, bb in zip(a, b)]) / len(a)


def top_k_accuracy(predictions, true_labels, k=3):
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
                top_k = list(labels.keys())[top_k[i, j]]
            except IndexError:
                top_k = "f"
            top_k_predictions[i, j] = top_k

    for square_ind in range(64):
        for j in range(k):
            if true_labels[square_ind] in top_k_predictions[square_ind, -j - 1 :]:
                hits[j] += 1

    accuracies = [hit / 64 for hit in hits]
    return tuple(accuracies)


def snake(squares):
    assert len(squares) == 64
    for i in range(8):
        squares[i * 8 : (i + 1) * 8] = list(reversed(squares[i * 8 : (i + 1) * 8]))
    squares = list(reversed(squares))
    return squares


def vectorize_chessboard(board):
    res = list("f" * 64)

    piecemap = board.piece_map()

    for i in range(64):
        piece = piecemap.get(i)
        if piece:
            res[i] = piece.symbol()

    return res


def get_test_generator(image_dir: Path):
    img_filenames = listdir_nohidden(image_dir)
    test_imgs = np.array([cv2.imread(str(image_dir / x)) for x in img_filenames])

    for i in range(len(test_imgs)):
        yield img_filenames[i], test_imgs[i]


def save_svg(chessboard: chess.Board, path: Path):
    svg = chess.svg.board(chessboard)
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(path))


def run_tests(
    image_folder: Path = TEST_DATA_DIR / "raw",
    truth_folder: Path = TEST_DATA_DIR / "ground_truth",
    run: tlc.Run | None = None,
    extractor: torch.nn.Module | None = None,
    classifier: torch.nn.Module | None = None,
    threshold=0.3,
    compute_metrics=True,
    create_table=False,
    project_name=PROJECT_NAME,
    dataset_name="dataset",
    table_name="table",
) -> tlc.Run:
    if not run:
        run = tlc.init(project_name)

    data_generator = get_test_generator(image_folder)

    table_writer: tlc.TableWriter | None = None

    if create_table:
        table_writer = tlc.TableWriter(
            table_name,
            dataset_name,
            project_name,
            column_schemas={
                "raw_img": tlc.PILImage("raw_img"),
            },
            if_exists="rename",
        )

    metrics_schemas = {
        "true_labels": tlc.CategoricalLabel("true_labels", LABEL_NAMES),
        "predicted_labels": tlc.CategoricalLabel("predicted_labels", LABEL_NAMES),
        "rendered_board": tlc.Schema(value=tlc.ImageUrlStringValue("rendered_board")),
    }
    if not table_writer:
        metrics_schemas = {"raw_img": tlc.PILImage("raw_img"), **metrics_schemas}

    metrics_writer = tlc.MetricsTableWriter(
        run.url,
        foreign_table_url=table_writer.url if table_writer else None,
        column_schemas=metrics_schemas,
    )

    top_1_accuracy = 0.0
    top_2_accuracy = 0.0
    top_3_accuracy = 0.0

    times = []
    test_set_size = len(listdir_nohidden(image_folder))

    with tlc.bulk_data_url_context(run.bulk_data_url, metrics_writer.url):
        for index, (filename, img) in tqdm(enumerate(data_generator), total=test_set_size, desc="Classifying images"):
            start = time.time()
            board_img, mask, predictions, chessboard, _, squares, _ = classify_raw(
                img,
                filename,
                extractor,
                classifier,
                threshold=threshold,
            )

            stop = time.time()
            times.append(stop - start)

            if board_img is None:
                print(f"Failed to classify {filename}")

                table_batch = {
                    "raw_img": [Image.open(str(image_folder / filename))],
                }

                metrics_batch = {
                    "predicted_masks": [Image.fromarray(mask.astype(np.uint8))],
                    "extracted_board": [BLACK_BOARD],
                    "rendered_board": [""],
                    "example_id": [index],
                    "is_failed": [True],
                }

                if table_writer:
                    table_writer.add_batch(table_batch)
                else:
                    metrics_batch["raw_img"] = table_batch["raw_img"]

                if compute_metrics:
                    metrics_batch["accuracy"] = [0]
                    metrics_batch["square_crop"] = [BLACK_CROP]
                    metrics_batch["true_labels"] = [0]
                    metrics_batch["predicted_labels"] = [0]

                metrics_writer.add_batch(metrics_batch)
                continue

            if compute_metrics:
                # Load the true labels
                truth_file = truth_folder / (filename[:-4] + ".txt")

                with open(truth_file) as truth:
                    true_labels = truth.read()

                true_labels = snake(list(true_labels))
                true_labels_int = [LABEL_NAMES.index(label) for label in true_labels]

                # Compute the accuracy
                top_1, top_2, top_3 = top_k_accuracy(predictions, true_labels, k=3)
                top_1_accuracy += top_1
                top_2_accuracy += top_2
                top_3_accuracy += top_3

                # Register the predicted labels
                predicted_labels = vectorize_chessboard(chessboard)
                predicted_labels = snake(predicted_labels)
                predicted_labels_int = [LABEL_NAMES.index(label) for label in predicted_labels]

                # Write and register a rendered board image
                svg_url = Path((run.bulk_data_url / "rendered_board" / (filename[:-4] + ".png")).to_str())
                svg_url.parent.mkdir(parents=True, exist_ok=True)
                save_svg(chessboard, svg_url)

                table_batch = {
                    "raw_img": [Image.open(str(image_folder / filename))] * 64,
                }
                metrics_batch = {
                    "predicted_masks": [Image.fromarray(mask.astype(np.uint8))] * 64,
                    "extracted_board": [Image.fromarray(board_img)] * 64,
                    "rendered_board": [str(svg_url)] * 64,
                    "accuracy": [top_1] * 64,
                    "square_crop": [Image.fromarray(img.squeeze()) for img in squares],
                    "true_labels": true_labels_int,
                    "predicted_labels": predicted_labels_int,
                    "example_id": [index] * 64,
                    "is_failed": [False] * 64,
                }
                if table_writer:
                    table_writer.add_batch(table_batch)
                else:
                    metrics_batch = {"raw_img": table_batch["raw_img"], **metrics_batch}

                metrics_writer.add_batch(metrics_batch)
            else:
                # Write and register a rendered board image
                svg_url = Path((run.bulk_data_url / "rendered_board" / (filename[:-4] + ".png")).to_str())
                svg_url.parent.mkdir(parents=True, exist_ok=True)
                save_svg(chessboard, svg_url)

                table_batch = {
                    "raw_img": [Image.open(str(image_folder / filename))],
                }

                metrics_batch = {
                    "predicted_masks": [Image.fromarray(mask.astype(np.uint8))],
                    "extracted_board": [Image.fromarray(board_img)],
                    "rendered_board": [str(svg_url)],
                    "example_id": [index],
                    "is_failed": [False],
                }

                if table_writer:
                    table_writer.add_batch(table_batch)
                else:
                    metrics_batch = {"raw_img": table_batch["raw_img"], **metrics_batch}

                metrics_writer.add_batch(metrics_batch)

    if compute_metrics:
        top_1_accuracy /= test_set_size
        top_2_accuracy /= test_set_size
        top_3_accuracy /= test_set_size

        aggregate_data = {
            "top_1_accuracy": f"{top_1_accuracy: .3f}",
            "top_2_accuracy": f"{top_2_accuracy: .3f}",
            "top_3_accuracy": f"{top_3_accuracy: .3f}",
            "avg_time_per_prediction": sum(times) / test_set_size,
        }

        run.set_parameters({"test_results": aggregate_data, "threshold": threshold})

    print(f"Classified {test_set_size} raw images")
    metrics_table = metrics_writer.finalize()
    if table_writer:
        print(f"Created table {table_writer.url}")
        table_writer.finalize()
    run.add_metrics_table(metrics_table)
    run.set_status_completed()

    return run


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--image-folder", type=str, default=str(TEST_DATA_DIR / "raw"))
    argparser.add_argument("--truth-folder", type=str, default=str(TEST_DATA_DIR / "ground_truth"))
    argparser.add_argument("--create-table", action="store_true")
    argparser.add_argument("--compute-metrics", action="store_true")
    argparser.add_argument("--threshold", type=float, default=0.3)
    argparser.add_argument("--project-name", type=str, default="chessvision-testing")
    argparser.add_argument("--dataset-name", type=str, default="dataset")
    argparser.add_argument("--table-name", type=str, default="table")

    return argparser.parse_args()


if __name__ == "__main__":
    print("Running ChessVision test suite...")
    args = parse_args()
    start = time.time()

    run = run_tests(
        image_folder=Path(args.image_folder),
        truth_folder=Path(args.truth_folder),
        compute_metrics=args.compute_metrics,
        create_table=args.create_table,
        project_name=args.project_name,
        dataset_name=args.dataset_name,
        table_name=args.table_name,
        threshold=args.threshold,
    )
    stop = time.time()
    print(f"Tests completed in {stop-start:.1f}s")
    if "test_results" in run.constants["parameters"]:
        print("Test accuracy: {}".format(run.constants["parameters"]["test_results"]["top_1_accuracy"]))
        print("Top-2 accuracy: {}".format(run.constants["parameters"]["test_results"]["top_2_accuracy"]))
        print("Top-3 accuracy: {}".format(run.constants["parameters"]["test_results"]["top_3_accuracy"]))

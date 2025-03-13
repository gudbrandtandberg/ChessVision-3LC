import argparse
import logging
from pathlib import Path
from typing import Any

import tlc
from ultralytics.utils.tlc import TLCYOLO, Settings

from chessvision import constants
from scripts.train import config
from scripts.train.create_classification_tables import get_or_create_tables

logger = logging.getLogger(__name__)


def train_model(
    model: TLCYOLO,
    epochs: int = 5,
    batch_size: int = -1,
    run_name: str = "",
    run_description: str = "",
    use_sample_weights: bool = True,
    patience: int = 5,
    train_table_name: str = "",
    val_table_name: str = "",
    collection_frequency: int = 1,
) -> Any:
    settings = Settings(
        project_name=config.YOLO_CLASSIFICATION_PROJECT,
        run_name=run_name,
        run_description=run_description,
        image_embeddings_dim=2,
        conf_thres=0.2,
        sampling_weights=use_sample_weights,
        exclude_zero_weight_training=True,
        exclude_zero_weight_collection=False,
        collection_epoch_start=1,
        collection_epoch_interval=collection_frequency,
    )

    tables = get_or_create_tables(
        train_table_name or config.INITIAL_TABLE_NAME,
        val_table_name or config.INITIAL_TABLE_NAME,
    )

    return model.train(
        tables=tables,
        settings=settings,
        batch=batch_size,
        imgsz=64,
        epochs=epochs,
        workers=4,
        project=str(Path(__file__).parent.parent.parent / "yolo_output"),
        patience=patience,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8m-cls.pt", help="YOLO model ID")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait before stopping")
    parser.add_argument("--batch-size", type=int, default=-1, help="Batch size")
    parser.add_argument("--run-name", type=str, default="", help="Run name")
    parser.add_argument("--run-description", type=str, default="", help="Run description")
    parser.add_argument("--use-sample-weights", action="store_true", help="Use sampling weights")
    parser.add_argument("--train-table-name", type=str, default="", help="Train table name")
    parser.add_argument("--val-table-name", type=str, default="", help="Val table name")
    parser.add_argument("--collection-frequency", type=int, default=1, help="Collection frequency")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    args = parse_args()

    model = TLCYOLO(args.model)

    results = train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        run_name=args.run_name,
        run_description=args.run_description,
        use_sample_weights=args.use_sample_weights,
        patience=args.patience,
        train_table_name=args.train_table_name,
        val_table_name=args.val_table_name,
        collection_frequency=args.collection_frequency,
    )

    classifier_checkpoint = results.save_dir / "weights" / "best.pt"

    if not Path(constants.BEST_YOLO_CLASSIFIER).exists():
        import shutil

        logger.info("Copying classifier checkpoint to %s", constants.BEST_YOLO_CLASSIFIER)
        shutil.copy(str(classifier_checkpoint), constants.BEST_YOLO_CLASSIFIER)

    if not args.skip_eval:
        from scripts.eval import evaluate_model

        logger.info("Running tests with trained model...")

        evaluate_model(
            run=tlc.active_run(),
            classifier_weights=str(classifier_checkpoint),
            classifier_model_id="yolo",
        )

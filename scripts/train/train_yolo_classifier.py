import argparse
import logging
from pathlib import Path
from typing import Any

import tlc
import torch
from tlc_ultralytics import YOLO, Settings

from chessvision import constants
from scripts.train import config
from scripts.train.create_classification_tables import get_or_create_tables
from scripts.utils import setup_logger

logger = logging.getLogger(__name__)


def metrics_collection_function(preds: Any, batch: Any) -> dict[str, Any]:
    # preds is already softmax-normalized for the classify task.
    top2 = torch.topk(preds, k=2, dim=1).values
    return {
        "top2_margin": top2[:, 0] - top2[:, 1],
        "entropy": -torch.sum(preds * torch.log(preds), dim=1),
    }


METRICS_SCHEMAS = {
    "top2_margin": tlc.schemas.Float32Schema(
        display_name="Top-2 Margin",
        description="P(top-1) - P(top-2). Low values indicate the model is on the fence between two classes.",
    ),
    "entropy": tlc.schemas.Float32Schema(
        display_name="Entropy",
        description="Shannon entropy of the predicted class distribution.",
    ),
}


def train_model(
    model: YOLO,
    *,
    epochs: int,
    batch_size: int,
    run_name: str,
    run_description: str,
    use_sample_weights: bool,
    patience: int,
    train_table_name: str,
    val_table_name: str,
    collection_frequency: int,
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
        image_embeddings_reducer_args={
            "retain_source_embedding_column": True,
            "delete_source_tables": False,
        },
        metrics_collection_function=metrics_collection_function,
        metrics_schemas=METRICS_SCHEMAS,
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
    args = parse_args()

    logger = setup_logger(__name__)

    logger.info("Running ChessVision training...")
    logger.info(f"Arguments: {args}")

    model = YOLO(args.model)

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
        Path(constants.BEST_YOLO_CLASSIFIER).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(classifier_checkpoint), constants.BEST_YOLO_CLASSIFIER)

    if not args.skip_eval:
        from scripts.eval import evaluate_model

        logger.info("Running tests with trained model...")

        evaluate_model(
            run=tlc.active_run(),
            classifier_weights=str(classifier_checkpoint),
            classifier_model_id="yolo",
            table_name="merged-2024-11-04-2024-11-04",
        )

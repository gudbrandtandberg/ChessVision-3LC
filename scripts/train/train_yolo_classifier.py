import argparse
import inspect
import logging
from pathlib import Path
from typing import Any

import tlc
import torch
from ultralytics.utils.tlc import TLCYOLO, Settings

from chessvision import constants
from scripts.train import config
from scripts.train.create_classification_tables import get_or_create_tables
from scripts.utils import setup_logger

logger = logging.getLogger(__name__)


def calculate_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Calculate entropy for classification probabilities.

    Args:
        probs: Probability tensor of shape [batch_size, num_classes]

    Returns:
        Entropy tensor of shape [batch_size]
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=1)
    return entropy


def create_entropy_logging_callbacks():
    """
    Create callbacks to log entropy metrics during validation.

    This replaces the deprecated metrics_collection_function parameter.
    Uses inspect to access validation predictions and logs aggregated entropy.
    """
    # Store entropy values for each validation batch
    entropy_values = []

    def on_val_batch_end(validator):
        """Callback triggered at the end of each validation batch."""
        try:
            # Use inspect to access batch and predictions from the calling frame
            frame = inspect.currentframe().f_back.f_back
            local_vars = frame.f_locals

            # Get predictions from the validator
            preds = local_vars.get("preds")

            if preds is not None and len(preds) > 0:
                # For classification, preds should contain probability distributions
                # Convert to tensor if needed
                if not isinstance(preds, torch.Tensor):
                    preds = torch.tensor(preds)

                # Apply softmax if not already probabilities
                if preds.dim() == 2:
                    probs = torch.softmax(preds, dim=1)
                else:
                    probs = preds

                # Calculate entropy for this batch
                batch_entropy = calculate_entropy(probs)
                entropy_values.extend(batch_entropy.cpu().tolist())

        except Exception as e:
            logger.debug(f"Could not calculate entropy for validation batch: {e}")

    def on_fit_epoch_end(trainer):
        """Callback triggered at the end of each epoch to log accumulated entropy."""
        if entropy_values:
            mean_entropy = sum(entropy_values) / len(entropy_values)

            # Log to 3LC
            tlc.log({
                "val_mean_entropy": mean_entropy,
                "epoch": trainer.epoch + 1,
            })

            logger.info(f"Epoch {trainer.epoch + 1} - Mean validation entropy: {mean_entropy:.4f}")

            # Clear for next epoch
            entropy_values.clear()

    return on_val_batch_end, on_fit_epoch_end


def train_model(
    model: TLCYOLO,
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
    )

    tables = get_or_create_tables(
        train_table_name or config.INITIAL_TABLE_NAME,
        val_table_name or config.INITIAL_TABLE_NAME,
    )

    # Add callbacks for entropy logging during validation
    val_batch_callback, epoch_end_callback = create_entropy_logging_callbacks()
    model.add_callback("on_val_batch_end", val_batch_callback)
    model.add_callback("on_fit_epoch_end", epoch_end_callback)

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
            table_name="merged-2024-11-04-2024-11-04",
        )

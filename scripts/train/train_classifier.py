from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import tlc
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import tqdm
from PIL.Image import Image
from torch.utils.data import DataLoader

from chessvision import constants, utils
from scripts.train import config
from scripts.train.create_classification_tables import get_or_create_tables
from scripts.train.training_utils import EarlyStopping, set_deterministic_mode, worker_init_fn
from scripts.utils import setup_logger

logger = logging.getLogger(__name__)

mp.set_start_method("spawn", force=True)

# Constants
LR_SCHEDULER_STEP_SIZE = 4
LR_SCHEDULER_GAMMA = 0.1
HIDDEN_LAYER_INDEX = 90

train_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=0),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.564], [0.246]),
    ],
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.564], [0.246]),
    ],
)


def train_map(sample: tuple[Image, int]) -> tuple[torch.Tensor, int]:
    return train_transforms(sample[0]), sample[1]


def val_map(sample: tuple[Image, int]) -> tuple[torch.Tensor, int]:
    return val_transforms(sample[0]), sample[1]


def train(
    model: nn.Module,
    train_loader: Any,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in tqdm.tqdm(train_loader, desc="Training", total=len(train_loader)):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target.to(device)).sum().item()

    train_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return train_loss, accuracy


def validate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm.tqdm(val_loader, desc="Validation", total=len(val_loader)):
            output = model(data.to(device))
            loss = criterion(output, target.to(device))
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target.to(device)).sum().item()

    val_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    return val_loss, accuracy


def save_classifier_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
) -> None:
    state_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": metadata,
    }
    torch.save(state_dict, checkpoint_path)


def train_model(
    model: torch.nn.Module,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    save_checkpoint: bool = True,
    run_name: str | None = None,
    run_description: str | None = None,
    use_sample_weights: bool = False,
    collection_frequency: int = 5,
    patience: int = 5,
    sweep_id: int | None = None,
    train_table_name: str = config.INITIAL_TABLE_NAME,
    val_table_name: str = config.INITIAL_TABLE_NAME,
    deterministic: bool = False,
    seed: int = 42,
) -> tuple[tlc.Run, tlc.Url]:
    if deterministic:
        seed_worker = worker_init_fn
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        seed_worker = None
        g = None

    # Create a 3LC Run
    parameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "sweep_id": sweep_id,
        "use_sample_weights": use_sample_weights,
        "learning_rate": learning_rate,
    }

    run = tlc.init(
        project_name=config.PIECE_CLASSIFICATION_PROJECT,
        run_name=run_name,
        description=run_description,
        parameters=parameters,
    )

    # Write checkpoints in the run directory
    checkpoint_path = run.bulk_data_url / "checkpoint.pth"
    checkpoint_path.make_parents(True)

    tables = get_or_create_tables(
        train_table_name=train_table_name,
        val_table_name=val_table_name,
    )

    train_table = tables["train"]
    val_table = tables["val"]

    train_table.map(train_map).map_collect_metrics(val_map)
    val_table.map(val_map)

    logger.info(f"Using training table {train_table.url}")
    logger.info(f"Using validation table {val_table.url}")

    # Create data loaders
    train_data_loader = DataLoader(
        train_table,
        batch_size=batch_size,
        shuffle=not use_sample_weights,
        sampler=train_table.create_sampler() if use_sample_weights else None,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_data_loader = DataLoader(
        val_table,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Set up metrics collection
    metrics_collectors = [
        tlc.ClassificationMetricsCollector(classes=constants.LABEL_NAMES),
        tlc.EmbeddingsMetricsCollector(layers=[HIDDEN_LAYER_INDEX]),
    ]

    predictor = tlc.Predictor(model, layers=[HIDDEN_LAYER_INDEX])

    # Set up training
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA)
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_accuracy = 0.0

    collection_epochs = list(range(collection_frequency, epochs + 1, collection_frequency))

    # Ensure at least one collection at the end
    if epochs not in collection_epochs:
        collection_epochs.append(epochs)

    # Train model
    start = time.time()
    for epoch in range(start_epoch, start_epoch + epochs):
        train_loss, train_acc = train(model, train_data_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_data_loader, criterion, device)
        scheduler.step()

        tlc.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
        )

        logger.info(
            f"Epoch: {epoch + 1}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, "
            f"Val Acc: {val_acc:.2f}%",
        )

        if val_acc > best_val_accuracy:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            logger.info("Saving model...")
            if save_checkpoint:
                save_classifier_checkpoint(
                    model,
                    checkpoint_path,
                    optimizer=optimizer,
                    metadata={
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                    },
                )

        if epoch in collection_epochs:
            tlc.collect_metrics(
                train_table,
                metrics_collectors=metrics_collectors,
                predictor=predictor,
                split="val",
                constants={"epoch": epoch},
                dataloader_args={"batch_size": 512},
                exclude_zero_weights=True,
                collect_aggregates=False,
            )

            tlc.collect_metrics(
                train_table,
                metrics_collectors=metrics_collectors,
                predictor=predictor,
                split="train",
                constants={"epoch": epoch},
                dataloader_args={"batch_size": 512},
                exclude_zero_weights=True,
                collect_aggregates=False,
            )

        early_stopping(val_loss)

        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

        # Clear cache between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # After training completes
    training_time = time.time() - start
    minutes = int(training_time // 60)
    seconds = int(training_time % 60)

    logger.info(f"Training completed in {minutes}m {seconds}s.")

    logger.info("Reducing embeddings to 2 dimensions using pacmap...")
    run.reduce_embeddings_by_foreign_table_url(
        train_table.url,
        method="pacmap",
        n_components=2,
        delete_source_tables=True,
    )

    run.set_parameters(
        {
            "best_val_accuracy": best_val_accuracy,
            "model_path": checkpoint_path.apply_aliases().to_str(),
            "train_table": tables["train"].url.apply_aliases().to_str(),
            "val_table": tables["val"].url.apply_aliases().to_str(),
            "final_epoch": epoch,
            "training_time": training_time,
        },
    )

    return run, checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--run-description", type=str, default="")
    parser.add_argument("--use-sample-weights", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--train-table", type=str, default=config.INITIAL_TABLE_NAME)
    parser.add_argument("--val-table", type=str, default=config.INITIAL_TABLE_NAME)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--sweep-id", type=int, default=None)
    parser.add_argument("--collection-frequency", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger = setup_logger(__name__)

    logger.info("Running ChessVision training...")
    logger.info(f"Arguments: {args}")

    if args.deterministic:
        set_deterministic_mode(args.seed)

    device = utils.get_device()
    logger.info(f"Using device {device}")

    model = utils.get_classifier_model()
    model = model.to(device)

    run, checkpoint_path = train_model(
        model=model,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        run_name=args.run_name,
        run_description=args.run_description,
        use_sample_weights=args.use_sample_weights,
        collection_frequency=args.collection_frequency,
        patience=args.patience,
        sweep_id=args.sweep_id,
        train_table_name=args.train_table,
        val_table_name=args.val_table,
    )

    if not Path(constants.BEST_CLASSIFIER_WEIGHTS).exists():
        import shutil

        logger.info("Copying classifier checkpoint to %s", constants.BEST_CLASSIFIER_WEIGHTS)
        shutil.copy(checkpoint_path.to_str(), constants.BEST_CLASSIFIER_WEIGHTS)

    if not args.skip_eval:
        from scripts.eval.evaluate import evaluate_model

        del model

        evaluate_model(
            run=run,
            classifier_weights=checkpoint_path.to_str(),
        )

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import timm
import tlc
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
from PIL.Image import Image
from torch.utils.data import DataLoader

from chessvision.core import ChessVision

from .training_utils import EarlyStopping

logger = logging.getLogger(__name__)

mp.set_start_method("spawn", force=True)

DATASET_ROOT = f"{ChessVision.DATA_ROOT}/squares"
tlc.register_url_alias("CHESSPIECES_DATASET_ROOT", DATASET_ROOT)
tlc.register_url_alias("PROJECT_ROOT", str(tlc.Configuration.instance().project_root_url))

TRAIN_DATASET_PATH = DATASET_ROOT + "/training"
VAL_DATASET_PATH = DATASET_ROOT + "/validation"

TRAIN_DATASET_NAME = "chesspieces-train"
VAL_DATASET_NAME = "chesspieces-val"

# Hyperparameters
NUM_CLASSES = ChessVision.NUM_CLASSES
BATCH_SIZE = 32
INITIAL_LR = 0.001
LR_SCHEDULER_STEP_SIZE = 4
LR_SCHEDULER_GAMMA = 0.1
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 4
HIDDEN_LAYER_INDEX = 90
MODEL_ID = ChessVision.MODEL_ID

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


train_dataset = datasets.ImageFolder(TRAIN_DATASET_PATH)
val_dataset = datasets.ImageFolder(VAL_DATASET_PATH)

sample_structure = (tlc.PILImage("image"), tlc.CategoricalLabel("label", classes=ChessVision.LABEL_NAMES))


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
        correct += predicted.eq(target).sum().item()

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
            correct += predicted.eq(target).sum().item()

    val_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    return val_loss, accuracy


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    filename: str = "checkpoint.pth",
) -> tuple[torch.nn.Module, torch.optim.Optimizer | None, int, float]:
    device = ChessVision.get_device()
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]
    return model, optimizer, epoch, best_val_loss


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int | None = None,
    best_val_loss: float | None = None,
    filename: str = "checkpoint.pth",
) -> str:
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )
    return filename


def main(args: argparse.Namespace) -> None:
    # Training variables
    start = time.time()
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)
    device = ChessVision.get_device()
    model = get_classifier_model()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA)
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_accuracy = 0.0

    # Create a Run object
    run_parameters = {
        "INITIAL_LR": INITIAL_LR,
        "LR_SCHEDULER_STEP_SIZE": LR_SCHEDULER_STEP_SIZE,
        "LR_SCHEDULER_GAMMA": LR_SCHEDULER_GAMMA,
        "MAX_EPOCHS": MAX_EPOCHS,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "MODEL_ID": MODEL_ID,
        "BATCH_SIZE": BATCH_SIZE,
    }

    run = tlc.init(
        project_name=args.project_name,
        run_name=args.run_name,
        description=args.description,
        parameters=run_parameters,
        if_exists="reuse" if args.resume else "rename",
    )

    checkpoint_path = run.bulk_data_url / "checkpoint.pth"
    checkpoint_path.make_parents(True)

    # Create datasets
    tlc_train_dataset = (
        tlc.Table.from_torch_dataset(
            dataset=train_dataset,
            dataset_name=TRAIN_DATASET_NAME,
            table_name="train",
            structure=sample_structure,
            project_name=args.project_name,
        )
        .map(train_map)
        .map_collect_metrics(val_map)
        .revision()
    )

    tlc_val_dataset = (
        tlc.Table.from_torch_dataset(
            dataset=val_dataset,
            dataset_name=VAL_DATASET_NAME,
            table_name="val",
            structure=sample_structure,
            project_name=args.project_name,
        )
        .map(val_map)
        .revision()
    )

    logger.info(f"Using training dataset: {tlc_train_dataset.url}")
    logger.info(f"Using validation dataset: {tlc_val_dataset.url}")

    # Create data loaders
    sampler = tlc_train_dataset.create_sampler() if args.use_sample_weights else None
    train_data_loader = DataLoader(
        tlc_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=not sampler,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_data_loader = DataLoader(
        tlc_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Set up metrics collection
    metrics_collection_dataloader_args = {
        "batch_size": 512,
    }
    chessvision_metrics_collector = tlc.ClassificationMetricsCollector(
        classes=ChessVision.LABEL_NAMES,
    )

    metrics_collectors = [
        chessvision_metrics_collector,
    ]
    if args.compute_embeddings:
        embeddings_collector = tlc.EmbeddingsMetricsCollector(layers=[HIDDEN_LAYER_INDEX])
        metrics_collectors.append(embeddings_collector)

    predictor = tlc.Predictor(model, layers=[HIDDEN_LAYER_INDEX])

    # Train model
    for epoch in range(start_epoch, start_epoch + MAX_EPOCHS):
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
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_val_loss,
                str(checkpoint_path),
            )

        early_stopping(val_loss)

        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    tlc.collect_metrics(
        tlc_val_dataset,
        metrics_collectors=metrics_collectors,
        predictor=predictor,
        split="val",
        constants={"epoch": epoch},
        dataloader_args=metrics_collection_dataloader_args,
        exclude_zero_weights=True,
        collect_aggregates=False,
    )

    tlc.collect_metrics(
        tlc_train_dataset,
        metrics_collectors=metrics_collectors,
        predictor=predictor,
        split="train",
        constants={"epoch": epoch},
        dataloader_args=metrics_collection_dataloader_args,
        exclude_zero_weights=True,
        collect_aggregates=False,
    )

    duration = time.time() - start
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    logger.info(f"Training completed in {minutes} minutes and {seconds} seconds.")

    if args.compute_embeddings:
        logger.info("Reducing embeddings...")
        run.reduce_embeddings_by_foreign_table_url(
            tlc_train_dataset.url,
            n_components=2,
            method="pacmap",
            delete_source_tables=True,
        )

    run.set_parameters(
        {
            "best_val_accuracy": best_val_accuracy,
            "model_path": checkpoint_path.apply_aliases().to_str(),
            "use_sample_weights": args.use_sample_weights,
        },
    )
    if args.run_tests:
        from chessvision.evaluate import evaluate_model

        logger.info("Running tests...")
        del model
        evaluate_model(run=run, classifier_weights=checkpoint_path.to_str())


def get_classifier_model() -> torch.nn.Module:
    """Initialize the piece classifier model."""
    return timm.create_model(MODEL_ID, num_classes=NUM_CLASSES, in_chans=1)  # type: ignore


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--project-name", type=str, default="chessvision-classification")
    argparser.add_argument("--run-name", type=str, default="")
    argparser.add_argument("--description", type=str, default="")
    argparser.add_argument("--compute-embeddings", action="store_true")
    argparser.add_argument("--resume", action="store_true")
    argparser.add_argument("--run-tests", action="store_true")
    argparser.add_argument("--use-sample-weights", action="store_true")
    args = argparser.parse_args()
    main(args)

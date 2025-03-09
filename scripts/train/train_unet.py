from __future__ import annotations

import argparse
import logging
import os
import random
import time
from typing import Any

import numpy as np
import tlc
import torch
import torch.nn as nn
import torchvision.transforms.functional as F  # noqa: N812
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as T  # noqa: N812
from tqdm import tqdm
from unet_loss_collector import LossCollector

from chessvision.core import ChessVision
from chessvision.pytorch_unet.evaluate import evaluate
from chessvision.pytorch_unet.unet import UNet
from chessvision.pytorch_unet.utils.dice_score import dice_loss

DATASET_ROOT = f"{ChessVision.DATA_ROOT}/board_extraction"
tlc.register_url_alias(
    "CHESSVISION_SEGMENTATION_DATA_ROOT",
    DATASET_ROOT,
)
tlc.register_url_alias(
    "CHESSVISION_SEGMENTATION_PROJECT_ROOT",
    f"{tlc.Configuration.instance().project_root_url}/chessvision-segmentation",
)


def worker_init_fn(worker_id: int) -> None:
    """Initialize worker with a unique seed."""
    worker_seed = 42 + worker_id  # Base seed + worker offset
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def set_deterministic_mode(seed: int = 42) -> None:
    """Set seeds and configurations for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def save_extractor_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    metadata: dict[str, Any],
) -> None:
    state_dict = {"model_state_dict": model.state_dict(), "metadata": metadata}
    torch.save(state_dict, checkpoint_path)


class TransformSampleToModel:
    """Convert a dict of PIL images to a dict of tensors of the right type."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        img = sample["image"]
        mask = sample["mask"]

        # Ensure consistent size
        if img.size != (256, 256):
            img = T.Resize((256, 256))(img)

        # Ensure mask is single channel
        if mask.mode != "L":
            mask = mask.convert("L")  # Convert to grayscale

        # Convert to tensors
        img_tensor = T.ToTensor()(img)
        mask_tensor = T.ToTensor()(mask).long()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
        }


class AugmentImages:
    """Apply random augmentations to the images and masks."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image, mask = sample["image"], sample["mask"]

        # Random horizontal flip
        if torch.rand(1) > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # Random rotation
        if torch.rand(1) > 0.5:
            angle = torch.randint(-15, 15, (1,)).item()
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)

        if torch.rand(1) > 0.5:
            image = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(image)

        if torch.rand(1) > 0.5:
            image = T.GaussianBlur(3)(image)

        return TransformSampleToModel()({"image": image, "mask": mask})


class PrepareModelOutputsForLogging:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(
        self,
        batch: dict[str, Any],
        predictor_output: tlc.PredictorOutput,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        predictions_tensor = predictor_output.forward

        for i in range(len(predictions_tensor)):
            predictions_tensor[i] = (torch.sigmoid(predictions_tensor[i]) > self.threshold) * 255

        return batch, predictions_tensor.squeeze(1)


def train_model(
    model: torch.nn.Module,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
    project_name: str = "chessvision-segmentation",
    run_name: str | None = None,
    run_description: str | None = None,
    use_sample_weights: bool = False,
    validations_per_epoch: int = 2,
    collection_frequency: int = 5,
    patience: int = 5,
    threshold: float = 0.5,
    seed: int = 42,
    deterministic: bool = False,
    sweep_id: int | None = None,
    train_table_name: str = "table",
    val_table_name: str = "table",
    train_dataset_name: str = "chessboard-segmentation-train",
    val_dataset_name: str = "chessboard-segmentation-val",
) -> tuple[tlc.Run, tlc.Url]:
    if deterministic:
        # Only set up worker seeds, main process already deterministic
        seed_worker = worker_init_fn
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        seed_worker = None
        g = None

    # Create 3LC datasets & training run
    parameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "sweep_id": sweep_id or 0,
        "use_sample_weights": use_sample_weights,
        "learning_rate": learning_rate,
    }
    run = tlc.init(
        project_name,
        run_name,
        parameters=parameters,
        description=run_description,
    )

    # Write checkpoints in the run directory
    checkpoint_path = run.bulk_data_url / "checkpoint.pth"
    checkpoint_path.make_parents(True)

    tlc_train_dataset = (
        tlc.Table.from_names(
            table_name=train_table_name,
            dataset_name=train_dataset_name,
            project_name=project_name,
        )
        .map(TransformSampleToModel())
        .revision()
    )

    tlc_val_dataset = (
        tlc.Table.from_names(
            table_name=val_table_name,
            dataset_name=val_dataset_name,
            project_name=project_name,
        )
        .map(TransformSampleToModel())
        .revision()
    )

    n_train = len(tlc_train_dataset)
    n_val = len(tlc_val_dataset)

    logging.info(f"Using training table {tlc_train_dataset.url}")
    logging.info(f"Using validation table {tlc_val_dataset.url}")

    # 3. Create data loaders
    train_loader = DataLoader(
        tlc_train_dataset,
        shuffle=not use_sample_weights,
        sampler=tlc_train_dataset.create_sampler() if use_sample_weights else None,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        tlc_val_dataset,
        shuffle=False,
        drop_last=True,
        batch_size=batch_size,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    logging.info(
        f"""Starting training:
        Epochs:           {epochs}
        Batch size:       {batch_size}
        Learning rate:    {learning_rate}
        Training size:    {n_train}
        Validation size:  {n_val}
        Checkpoints:      {save_checkpoint}
        Device:           {device.type}
        Images scaling:   {img_scale}
        Mixed Precision:  {amp}
        Sampling weights: {use_sample_weights}
    """,
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=3)  # goal: maximize Dice score
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.BCEWithLogitsLoss()
    global_step = 0
    best_val_score = float("-inf")
    patience_counter = 0

    # Set model memory format
    model = model.to(memory_format=torch.channels_last)  # type: ignore

    # Calculate validation interval based on epochs rather than steps
    total_steps = n_train // batch_size
    validation_interval = total_steps // validations_per_epoch

    # Calculate collection epochs dynamically
    collection_epochs = list(range(collection_frequency, epochs + 1, collection_frequency))

    # Ensure at least one collection at the end
    if epochs not in collection_epochs:
        collection_epochs.append(epochs)

    # Add training config to metadata
    training_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "amp": amp,
        "threshold": threshold,
        "run_name": run.name,
    }

    # Save initial model state
    save_extractor_checkpoint(
        model,
        checkpoint_path,
        metadata={
            "best_val_score": float("-inf"),
            "training_config": training_config,
            "epoch": 0,
        },
    )

    # 5. Begin training
    start_time = time.time()  # Start timing at beginning of training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for i, batch in enumerate(train_loader):
                images, true_masks = batch["image"], batch["mask"]

                assert images.shape[1] == model.n_channels, (
                    f"Network has been defined with {model.n_channels} input channels, "
                    f"but loaded images have {images.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )

                # Ensure inputs match model's memory format
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        torch.sigmoid(masks_pred),
                        true_masks,
                        multiclass=False,
                        reduce_batch_first=False,
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{"loss (batch)": loss.item()})

                # Run validation based on interval
                if i > 0 and i % validation_interval == 0:
                    val_score = evaluate(model, val_loader, device, amp)
                    scheduler.step(val_score)

                    tlc.log(
                        {
                            "val_dice": val_score.item(),
                            "step": global_step,
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                    )

        if save_checkpoint and val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            save_extractor_checkpoint(
                model,
                checkpoint_path,
                metadata={
                    "best_val_score": best_val_score,
                    "epoch": epoch,
                    "training_config": training_config,
                },
            )
            logging.info(f"Checkpoint {epoch} saved! (Dice score: {best_val_score}, run: {run.name})")
        else:
            patience_counter += 1

        tlc.log({"train_loss": epoch_loss / n_train, "epoch": epoch})

        # Collect per-sample metrics using tlc every 5 epochs
        if epoch in collection_epochs:
            predictor = tlc.Predictor(
                model=model,
                layers=[52],
            )

            collectors = [
                LossCollector(),
                tlc.SegmentationMetricsCollector(
                    label_map=ChessVision.SEGMENTATION_MAP,
                    threshold=threshold,
                ),
                tlc.EmbeddingsMetricsCollector(layers=[52], reshape_strategy={52: "mean"}),
            ]

            tlc.collect_metrics(
                tlc_train_dataset,
                metrics_collectors=collectors,
                predictor=predictor,
                split="train",
                constants={"step": global_step, "epoch": epoch},
                dataloader_args={"batch_size": 4},
            )
            tlc.collect_metrics(
                tlc_val_dataset,
                metrics_collectors=collectors,
                predictor=predictor,
                split="val",
                constants={"step": global_step, "epoch": epoch},
                dataloader_args={"batch_size": 4},
            )

            if patience_counter >= patience and epoch != epochs:
                logging.info(f"Early stopping triggered after {epoch} epochs")
                break

        # Clear cache between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # After training completes
    stop_time = time.time()
    training_minutes = int((stop_time - start_time) // 60)
    training_seconds = int((stop_time - start_time) % 60)
    training_time = f"{training_minutes}m {training_seconds}s"

    logging.info(f"Training completed in {training_time}")

    logging.info("Reducing embeddings to 2 dimensions using pacmap...")
    run.reduce_embeddings_by_foreign_table_url(
        tlc_train_dataset.url,
        delete_source_tables=True,
        method="pacmap",
        n_components=2,
    )

    run.set_parameters(
        {
            "best_val_score": best_val_score,
            "model_path": checkpoint_path.apply_aliases().to_str(),
            "train_table": tlc_train_dataset.url.apply_aliases().to_str(),
            "val_table": tlc_val_dataset.url.apply_aliases().to_str(),
            "final_epoch": epoch,
            "training_time": training_time,
        },
    )
    return run, checkpoint_path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")
    parser.add_argument("--epochs", "-e", metavar="E", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", dest="batch_size", metavar="B", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--learning-rate",
        "-l",
        metavar="LR",
        type=float,
        default=1e-7,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument("--load", "-f", type=str, default=False, help="Load model from a .pth file")
    parser.add_argument("--scale", "-s", type=float, default=1.0, help="Downscaling factor of the images")
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision")
    parser.add_argument("--bilinear", action="store_true", default=False, help="Use bilinear upsampling")
    parser.add_argument("--classes", "-c", type=int, default=1, help="Number of classes")
    parser.add_argument("--run-tests", action="store_true", help="Run the test suite after training")
    parser.add_argument("--project-name", type=str, default="chessvision-segmentation", help="3LC project name")
    parser.add_argument("--run-name", type=str, default=None, help="3LC run name")
    parser.add_argument("--run-description", type=str, default=None, help="3LC run description")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing the output masks")
    parser.add_argument("--use-sample-weights", action="store_true", help="Use a weighted sampler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training")
    parser.add_argument("--sweep-id", type=int, default=None, help="3LC sweep ID")
    parser.add_argument("--train-table", type=str, default="table", help="Name of training table")
    parser.add_argument("--val-table", type=str, default="table", help="Name of validation table")
    parser.add_argument(
        "--train-dataset",
        type=str,
        default="chessboard-segmentation-train",
        help="Name of training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        type=str,
        default="chessboard-segmentation-val",
        help="Name of validation dataset",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Set deterministic mode BEFORE creating model
    if args.deterministic:
        set_deterministic_mode(args.seed)

    device = ChessVision.get_device()
    logging.info(f"Using device {device}")

    model: torch.nn.Module = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model.to(device=device)

    logging.info(
        f"Network:\n"
        f"\t{model.n_channels} input channels\n"
        f"\t{model.n_classes} output channels (classes)\n"
        f"\t{'Bilinear' if model.bilinear else 'Transposed conv'} upscaling"
        f"\t{'Device: ' + str(device)}",
    )

    if args.load:
        model = ChessVision.load_model_checkpoint(model, ChessVision.EXTRACTOR_WEIGHTS, device)

    run, checkpoint_path = train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        amp=args.amp,
        project_name=args.project_name,
        run_name=args.run_name,
        run_description=args.run_description,
        use_sample_weights=args.use_sample_weights,
        validations_per_epoch=2,
        collection_frequency=5,
        patience=5,
        threshold=args.threshold,
        seed=args.seed,
        deterministic=args.deterministic,
        sweep_id=args.sweep_id,
        train_table_name=args.train_table,
        val_table_name=args.val_table,
        train_dataset_name=args.train_dataset,
        val_dataset_name=args.val_dataset,
    )

    # Run evaluation if requested
    if args.run_tests:
        from chessvision.evaluate import evaluate_model

        del model

        evaluate_model(
            run=run,
            threshold=args.threshold,
            board_extractor_weights=str(checkpoint_path),
        )

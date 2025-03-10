from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import tlc
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter, GaussianBlur, Resize, ToTensor
from torchvision.transforms.functional import hflip, rotate
from tqdm import tqdm

from chessvision import constants, utils
from chessvision.pytorch_unet.evaluate import evaluate
from chessvision.pytorch_unet.unet import UNet
from chessvision.pytorch_unet.utils.dice_score import dice_loss
from scripts.train import config
from scripts.train.create_board_extraction_tables import get_or_create_tables
from scripts.train.training_utils import set_deterministic_mode, worker_init_fn
from scripts.train.unet_loss_collector import LossCollector

logger = logging.getLogger(__name__)


def save_extractor_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    metadata: dict[str, Any],
) -> None:
    state_dict = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(state_dict, checkpoint_path)


class TransformSampleToModel:
    """Convert a dict of PIL images to a dict of tensors of the right type."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        img = sample["image"]
        mask = sample["mask"]

        # Ensure consistent size
        if img.size != (256, 256):
            img = Resize((256, 256))(img)

        # Ensure mask is single channel
        if mask.mode != "L":
            mask = mask.convert("L")  # Convert to grayscale

        # Convert to tensors
        img_tensor = ToTensor()(img)
        mask_tensor = ToTensor()(mask).long()

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
            image = hflip(image)
            mask = hflip(mask)

        # Random rotation
        if torch.rand(1) > 0.5:
            angle = torch.randint(-15, 15, (1,)).item()
            image = rotate(image, angle)
            mask = rotate(mask, angle)

        if torch.rand(1) > 0.5:
            image = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(image)

        if torch.rand(1) > 0.5:
            image = GaussianBlur(3)(image)

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
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
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
    train_table_name: str = config.INITIAL_TABLE_NAME,
    val_table_name: str = config.INITIAL_TABLE_NAME,
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
        "sweep_id": sweep_id or 0,
        "use_sample_weights": use_sample_weights,
        "learning_rate": learning_rate,
    }
    run = tlc.init(
        project_name=config.BOARD_EXTRACTION_PROJECT,
        run_name=run_name,
        parameters=parameters,
        description=run_description,
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

    train_table.map(AugmentImages()).map_collect_metrics(TransformSampleToModel())
    val_table.map(TransformSampleToModel())

    n_train = len(train_table)
    n_val = len(val_table)

    logger.info(f"Using training table {train_table.url}")
    logger.info(f"Using validation table {val_table.url}")

    # Create data loaders
    train_loader = DataLoader(
        train_table,
        shuffle=not use_sample_weights,
        sampler=train_table.create_sampler() if use_sample_weights else None,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_table,
        shuffle=False,
        drop_last=True,
        batch_size=batch_size,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Set up metrics collection
    predictor = tlc.Predictor(
        model=model,
        layers=[52],
    )

    metrics_collectors = [
        LossCollector(),
        tlc.SegmentationMetricsCollector(
            label_map=constants.SEGMENTATION_MAP,
            preprocess_fn=PrepareModelOutputsForLogging(threshold=threshold),
        ),
        tlc.EmbeddingsMetricsCollector(layers=[52]),
    ]

    logger.info(
        f"""Starting training:
        Epochs:           {epochs}
        Batch size:       {batch_size}
        Learning rate:    {learning_rate}
        Training size:    {n_train}
        Validation size:  {n_val}
        Checkpoints:      {save_checkpoint}
        Device:           {device.type}
        Mixed Precision:  {amp}
        Sampling weights: {use_sample_weights}
    """,
    )

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
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
    start_time = time.time()
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
            logger.info(f"Checkpoint {epoch} saved! (Dice score: {best_val_score}, run: {run.name})")
        else:
            patience_counter += 1

        tlc.log(
            {
                "train_loss": epoch_loss / n_train,
                "epoch": epoch,
            },
        )

        # Collect per-sample metrics using tlc every 5 epochs
        if epoch in collection_epochs:
            tlc.collect_metrics(
                train_table,
                metrics_collectors=metrics_collectors,
                predictor=predictor,
                split="train",
                constants={"step": global_step, "epoch": epoch},
                dataloader_args={"batch_size": 4},
            )
            tlc.collect_metrics(
                val_table,
                metrics_collectors=metrics_collectors,
                predictor=predictor,
                split="val",
                constants={"step": global_step, "epoch": epoch},
                dataloader_args={"batch_size": 4},
            )

            if patience_counter >= patience and epoch != epochs:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        # Clear cache between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # After training completes
    training_time = time.time() - start_time
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
            "best_val_score": best_val_score,
            "model_path": checkpoint_path.apply_aliases().to_str(),
            "train_table": train_table.url.apply_aliases().to_str(),
            "val_table": val_table.url.apply_aliases().to_str(),
            "final_epoch": epoch,
            "training_time": training_time,
        },
    )
    return run, checkpoint_path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")
    parser.add_argument("--run-name", type=str, default=None, help="3LC run name")
    parser.add_argument("--run-description", type=str, default=None, help="3LC run description")
    parser.add_argument("--run-tests", action="store_true", help="Run the test suite after training")
    parser.add_argument("--use-sample-weights", action="store_true", help="Use a weighted sampler")
    parser.add_argument("--train-table", type=str, default=config.INITIAL_TABLE_NAME, help="Name of training table")
    parser.add_argument("--val-table", type=str, default=config.INITIAL_TABLE_NAME, help="Name of validation table")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training")
    parser.add_argument("--sweep-id", type=int, default=None, help="3LC sweep ID")
    parser.add_argument("--collection-frequency", type=int, default=5, help="Number of epochs between collections")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait before early stopping")
    parser.add_argument("--load", "-f", type=str, default=False, help="Load model from a .pth file")
    parser.add_argument("--scale", "-s", type=float, default=1.0, help="Downscaling factor of the images")
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision")
    parser.add_argument("--bilinear", action="store_true", default=False, help="Use bilinear upsampling")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing the output masks")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.deterministic:
        set_deterministic_mode(args.seed)

    device = utils.get_device()
    logger.info(f"Using device {device}")

    model: torch.nn.Module = UNet(
        n_channels=3,
        n_classes=1,
        bilinear=args.bilinear,
    )
    model.to(device=device)

    if args.load:
        model = utils.load_model_checkpoint(
            model,
            constants.BEST_EXTRACTOR_WEIGHTS,
            device,
        )

    run, checkpoint_path = train_model(
        model=model,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        amp=args.amp,
        run_name=args.run_name,
        run_description=args.run_description,
        use_sample_weights=args.use_sample_weights,
        validations_per_epoch=2,
        collection_frequency=args.collection_frequency,
        patience=args.patience,
        threshold=args.threshold,
        seed=args.seed,
        deterministic=args.deterministic,
        sweep_id=args.sweep_id,
        train_table_name=args.train_table,
        val_table_name=args.val_table,
    )

    if args.run_tests:
        from chessvision.evaluate import evaluate_model

        del model

        evaluate_model(
            run=run,
            threshold=args.threshold,
            board_extractor_weights=checkpoint_path.to_str(),
        )

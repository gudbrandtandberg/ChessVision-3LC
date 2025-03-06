from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import tlc
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from chessvision.board_extraction.loss_collector import LossCollector
from chessvision.predict.classify_raw import load_extractor_checkpoint
from chessvision.pytorch_unet.evaluate import evaluate
from chessvision.pytorch_unet.unet import UNet
from chessvision.pytorch_unet.utils.dice_score import dice_loss
from chessvision.utils import DATA_ROOT, best_extractor_weights, get_device, segmentation_map

DATASET_ROOT = f"{DATA_ROOT}/board_extraction"
tlc.register_url_alias(
    "CHESSVISION_SEGMENTATION_DATA_ROOT",
    DATASET_ROOT,
)
tlc.register_url_alias(
    "CHESSVISION_SEGMENTATION_PROJECT_ROOT",
    f"{tlc.Configuration.instance().project_root_url}/chessvision-segmentation",
)


def save_extractor_checkpoint(model: torch.nn.Module, checkpoint_path: str, metadata: dict[str, Any]):
    state_dict = {"model_state_dict": model.state_dict(), "metadata": metadata}
    torch.save(state_dict, checkpoint_path)


class TransformSampleToModel:
    """Convert a dict of PIL images to a dict of tensors of the right type."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            "image": T.ToTensor()(sample["image"]),
            "mask": T.ToTensor()(sample["mask"]).long(),
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
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def __call__(self, batch, predictor_output: tlc.PredictorOutput):
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
    use_sample_weights: bool = False,
    validations_per_epoch: int = 2,
    collection_frequency: int = 5,
    patience: int = 5,
    threshold: float = 0.3,
):
    # Create 3LC datasets & training run
    parameters = {
        "epochs": epochs,
        "batch_size": batch_size,
    }
    run = tlc.init(project_name, run_name, parameters=parameters)

    # Write checkpoints in the run directory
    checkpoint_path = run.bulk_data_url / "checkpoint.pth"
    checkpoint_path.make_parents(True)

    tlc_train_dataset = (
        tlc.Table.from_names(
            "train-cleaned-filtered",
            "chessboard-segmentation-train",
            project_name,
        )
        .map(TransformSampleToModel())
        .revision()
    )

    tlc_val_dataset = (
        tlc.Table.from_names(
            "table",
            "chessboard-segmentation-val",
            project_name,
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
        shuffle=False if use_sample_weights else True,
        sampler=tlc_train_dataset.create_sampler() if use_sample_weights else None,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        tlc_val_dataset,
        shuffle=False,
        drop_last=True,
        batch_size=batch_size,
        pin_memory=True,
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
    """
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
    model = model.to(memory_format=torch.channels_last)

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
    }

    # Save initial model state
    save_extractor_checkpoint(
        model, checkpoint_path, {"epoch": 0, "best_val_score": float("-inf"), "training_config": training_config}
    )

    # 5. Begin training
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
                        }
                    )

        if save_checkpoint and val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            save_extractor_checkpoint(
                model,
                checkpoint_path,
                metadata={
                    "best_val_score": best_val_score,
                    "run_name": run.name,
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
                    label_map=segmentation_map,
                    preprocess_fn=PrepareModelOutputsForLogging(),
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
                print(f"Early stopping triggered after {epoch} epochs")
                break

        # Clear cache between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info("Training completed. Reducing embeddings to 2 dimensions using pacmap...")
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
            "use_sample_weights": use_sample_weights,
            "train_table": tlc_train_dataset.url.apply_aliases().to_str(),
            "val_table": tlc_val_dataset.url.apply_aliases().to_str(),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
        }
    )
    return run, checkpoint_path


def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")
    parser.add_argument("--epochs", "-e", metavar="E", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", dest="batch_size", metavar="B", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--learning-rate", "-l", metavar="LR", type=float, default=1e-7, help="Learning rate", dest="lr"
    )
    parser.add_argument("--load", "-f", type=str, default=False, help="Load model from a .pth file")
    parser.add_argument("--scale", "-s", type=float, default=1.0, help="Downscaling factor of the images")
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision")
    parser.add_argument("--bilinear", action="store_true", default=False, help="Use bilinear upsampling")
    parser.add_argument("--classes", "-c", type=int, default=1, help="Number of classes")
    parser.add_argument("--run-tests", action="store_true", help="Run the test suite after training")
    parser.add_argument("--project-name", type=str, default="chessvision-segmentation", help="3LC project name")
    parser.add_argument("--run-name", type=str, default=None, help="3LC run name")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold for binarizing the output masks")
    parser.add_argument("--use-sample-weights", action="store_true", help="Use a weighted sampler")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = get_device()
    logging.info(f"Using device {device}")

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(
        f"Network:\n"
        f"\t{model.n_channels} input channels\n"
        f"\t{model.n_classes} output channels (classes)\n"
        f"\t{'Bilinear' if model.bilinear else 'Transposed conv'} upscaling"
        f"\t{'Device: ' + str(device)}"
    )

    if args.load:
        load_extractor_checkpoint(model, best_extractor_weights)

    model.to(device=device)

    start = time.time()
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
        use_sample_weights=args.use_sample_weights,
        validations_per_epoch=2,
        collection_frequency=5,
        patience=5,
        threshold=args.threshold,
    )
    stop = time.time()
    minutes = int((stop - start) // 60)
    seconds = int((stop - start) % 60)
    print(f"Training completed in {minutes}m {seconds}s")

    if args.run_tests:
        from chessvision.test import run_tests

        del model

        model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        model = model.to(memory_format=torch.channels_last)
        model = load_extractor_checkpoint(model, checkpoint_path)
        model.to(device=device)

        run_tests(run=run, extractor=model, threshold=args.threshold)

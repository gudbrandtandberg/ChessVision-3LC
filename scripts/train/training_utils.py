import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = True, delta: float = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = float("inf")
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss: float) -> None:
        score = -val_loss

        if self.best_score == float("inf"):
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def plot_metrics(
    train_losses: list[float],
    val_losses: list[float],
    train_accuracies: list[float],
    val_accuracies: list[float],
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    input("Press any key to continue...")


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

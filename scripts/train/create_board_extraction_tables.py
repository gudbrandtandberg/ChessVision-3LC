from pathlib import Path

import tlc
import torch
from torch.utils.data import random_split

from chessvision.core import ChessVision
from chessvision.pytorch_unet.utils.data_loading import BasicDataset

DATASET_ROOT = f"{ChessVision.DATA_ROOT}/board_extraction"
tlc.register_url_alias(
    "CHESSVISION_SEGMENTATION_DATA_ROOT",
    DATASET_ROOT,
)
tlc.register_url_alias(
    "CHESSVISION_SEGMENTATION_PROJECT_ROOT",
    f"{tlc.Configuration.instance().project_root_url}/chessvision-segmentation",
)

dir_img = Path(DATASET_ROOT) / "images/"
dir_mask = Path(DATASET_ROOT) / "masks/"
assert dir_img.exists()
assert dir_mask.exists()


def create_tables(val_percent: float = 0.1) -> dict[str, tlc.Table]:
    # 1. Create dataset
    dataset = BasicDataset(dir_img.as_posix(), dir_mask.as_posix(), scale=1.0)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    sample_structure = {
        "image": tlc.PILImage("image"),
        "mask": tlc.SegmentationPILImage("mask", classes=ChessVision.SEGMENTATION_MAP),
    }

    tlc_val_dataset = tlc.Table.from_torch_dataset(
        dataset=val_set,
        dataset_name="chessboard-segmentation-val",
        structure=sample_structure,
        if_exists="reuse",
    )

    tlc_train_dataset = tlc.Table.from_torch_dataset(
        dataset=train_set,
        dataset_name="chessboard-segmentation-train",
        structure=sample_structure,
        if_exists="reuse",
    )

    return {
        "val": tlc_val_dataset,
        "train": tlc_train_dataset,
    }


if __name__ == "__main__":
    create_tables()

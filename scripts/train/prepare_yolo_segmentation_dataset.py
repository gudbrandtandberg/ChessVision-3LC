"""Build a YOLO-segmentation dataset for board extraction, honoring the canonical 3LC split.

Train/val membership is taken from the `chessvision-segmentation` tables
(`scripts/train/create_board_extraction_tables.py` — a seeded 10% `random_split`), NOT a fresh
split, so the YOLO-seg dataset stays consistent with every other use of the board-extraction
data. Masks are remapped to a single class (board=1, background=0) and converted to YOLO
polygon labels; a `data.yaml` is written for `train_yolo_segmentation_model.py`.

Usage:
    python scripts/train/prepare_yolo_segmentation_dataset.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import tlc
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

from chessvision import constants
from scripts.train import config  # noqa: F401  imported for its URL-alias registration side effects
from scripts.train.create_board_extraction_tables import get_or_create_tables

OUT = constants.DATA_ROOT / "board_extraction" / "yolo"


def main() -> None:
    tables = get_or_create_tables(config.INITIAL_TABLE_NAME, config.INITIAL_TABLE_NAME)
    # Rebuild from scratch so a previous (differently-split) dataset can't leave stale files.
    shutil.rmtree(OUT, ignore_errors=True)

    for split in ("train", "val"):
        table = tables[split]
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "masks" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for row in table.table_rows:
            img_path = Path(tlc.Url(row["image"]).to_absolute().to_str())
            mask_path = Path(tlc.Url(row["mask"]).to_absolute().to_str())
            shutil.copy(img_path, OUT / "images" / split / img_path.name)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(OUT / "masks" / split / (img_path.stem + ".png")), (mask > 0).astype(np.uint8))
        # convert_segment_masks_to_yolo_seg does not create the output dir itself.
        convert_segment_masks_to_yolo_seg(str(OUT / "masks" / split), str(OUT / "labels" / split), classes=1)
        print(f"{split}: {len(table)} samples")

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: chessboard\n",
    )
    print(f"wrote {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()

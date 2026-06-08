"""Build a YOLO-segmentation dataset for board extraction from the raw image/mask folders.

Reads `data/board_extraction/{images,masks}` (parallel-named `<stem>.JPG` / `<stem>.png`,
masks binary 0/255), makes a deterministic train/val split, remaps masks to a single
class (board=1, background=0), converts them to YOLO polygon labels, and writes a
`data.yaml`. The result under `data/board_extraction/yolo/` is what
`scripts/train/train_yolo_segmentation_model.py` ingests on first run.

Usage:
    python scripts/train/prepare_yolo_segmentation_dataset.py [--val-fraction 0.1]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

from chessvision import constants

ROOT = constants.DATA_ROOT / "board_extraction"
OUT = ROOT / "yolo"


def main(val_fraction: float) -> None:
    imgs = sorted((ROOT / "images").glob("*.JPG"))
    if not imgs:
        raise FileNotFoundError(f"No images found in {ROOT / 'images'}")
    n_val = max(1, int(len(imgs) * val_fraction))
    splits = {"val": imgs[:n_val], "train": imgs[n_val:]}
    print(f"total={len(imgs)} train={len(splits['train'])} val={len(splits['val'])}")

    for split, files in splits.items():
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "masks" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy(img, OUT / "images" / split / img.name)
            mask = cv2.imread(str(ROOT / "masks" / (img.stem + ".png")), cv2.IMREAD_GRAYSCALE)
            # Single class: board pixels -> 1, background -> 0.
            cv2.imwrite(str(OUT / "masks" / split / (img.stem + ".png")), (mask > 0).astype(np.uint8))
        # convert_segment_masks_to_yolo_seg does not create the output dir itself.
        convert_segment_masks_to_yolo_seg(str(OUT / "masks" / split), str(OUT / "labels" / split), classes=1)

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: chessboard\n",
    )
    print(f"wrote {OUT / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    main(parser.parse_args().val_fraction)

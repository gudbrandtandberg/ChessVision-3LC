from pathlib import Path

import cv2
import tlc

from chessvision.core import ChessVision
from scripts.train.config import BOARD_EXTRACTION_ROOT

kpt_names = ["top_left", "top_right", "bottom_right", "bottom_left"]
lines = [0, 1, 1, 2, 2, 3, 3, 0]

column_schemas = {
    "image": tlc.ImagePath("image"),
    "label": tlc.Keypoints2DSchema(
        keypoint_shape=(4, 2),
        keypoint_names=kpt_names,
        lines_default_value=lines,
        relative=False,
        writable=True,
    ),
}

if __name__ == "__main__":
    table_data = {
        "image": [],
        "label": [],
    }

    masks_path = Path(BOARD_EXTRACTION_ROOT) / "masks"
    for mask_path in masks_path.glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape
        assert h == w, "Mask must be square"
        assert h == 256, "Mask must be 256x256"
        quadrangle = ChessVision._find_quadrangle(mask)
        if quadrangle is None:
            print(f"No quadrangle found for {mask_path}")
            continue

        xys = quadrangle.reshape(-1)  # / 256.0
        image_path = mask_path.parent.parent / "images" / (mask_path.stem + ".jpg")

        table_data["image"].append(str(image_path))

        label = {
            "instances": [
                {
                    "xys": xys,
                    "lines": lines,
                },
            ],
            "x_min": 0,
            "y_min": 0,
            "x_max": 256,
            "y_max": 256,
        }
        table_data["label"].append(label)
        # break

    table = tlc.Table.from_dict(
        data=table_data,
        structure=column_schemas,
        table_name="chessvision-kpts-test",
        dataset_name="chessvision-kpts-test",
        project_name="chessvision-kpts",
        if_exists="rename",
    )

    print(table)

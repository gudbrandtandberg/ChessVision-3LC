"""Script that takes a batch of new test images (as a tlc.Table),
and copies the image file and the FEN to the test data directory."""

import shutil
from pathlib import Path

import numpy as np
import tlc

from chessvision import ChessVision, constants

if __name__ == "__main__":
    table_name = "delete-bad-sample"
    dataset_name = "test"
    project_name = "chessvision-testing"

    test_folder_name = "2024-11-04-2024-11-04"
    table = tlc.Table.from_names(table_name, dataset_name, project_name)

    cv = ChessVision(lazy_load=False)

    for img in table:
        image_url = Path(img._tlc_url)
        # img is Image.Image object
        img_np = np.array(img)
        if image_url.parent != constants.DATA_ROOT / "test" / "raw":
            target_path = constants.DATA_ROOT / "test" / test_folder_name / "raw" / image_url.name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(image_url, target_path)

            result = cv.process_image(img_np)
            target_ground_truth_path = (
                constants.DATA_ROOT / "test" / test_folder_name / "ground_truth" / (image_url.stem + ".txt")
            )
            target_ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
            with target_ground_truth_path.open("w") as f:
                f.write(result.position.fen) if result.position else f.write("")
        else:
            print(f"Skipping {image_url} because it already exists")

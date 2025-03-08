import shutil
from pathlib import Path

from ultralytics.data.converter import convert_segment_masks_to_yolo_seg


def make_splits():
    import tlc

    val_table = tlc.Table.from_url(
        "C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/chessvision-segmentation/datasets/chessboard-segmentation-val/tables/table_0000",
    )
    train_table = tlc.Table.from_url(
        "C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/chessvision-segmentation/datasets/chessboard-segmentation-train/tables/table_0000",
    )

    root_dir = "data/board_extraction/yolo/masks"
    for split, table in [("val", val_table), ("train", train_table)]:
        Path(f"{root_dir}/{split}").mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(table.table_rows):
            image_url = tlc.Url(row["mask"]).to_absolute()
            assert image_url.exists(), f"Image {image_url} does not exist"
            shutil.copy(image_url.to_str(), f"{root_dir}/{split}")
            assert True


if __name__ == "__main__":
    # make_splits()

    convert_segment_masks_to_yolo_seg(
        "data/board_extraction/yolo/masks/train", "data/board_extraction/yolo/labels/train", classes=255,
    )
    convert_segment_masks_to_yolo_seg(
        "data/board_extraction/yolo/masks/val", "data/board_extraction/yolo/labels/val", classes=255,
    )

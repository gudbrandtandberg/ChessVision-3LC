from pathlib import Path

# import tlc
# import torch
# from ultralytics.utils.tlc import TLCYOLO, Settings
from ultralytics import YOLO

# from chessvision.test import run_tests

if __name__ == "__main__":
    model = YOLO("yolo11m-seg.pt")

    # settings = Settings(
    #     project_name="chessvision-classification",
    #     run_description="Train on cleaned training and val set",
    #     image_embeddings_dim=2,
    #     conf_thres=0.2,
    #     sampling_weights=True,
    #     exclude_zero_weight_training=True,
    #     exclude_zero_weight_collection=False,
    # )
    results = model.train(
        # tables={
        #     "train": tlc.Url.create_table_url("train-cleaned", "chesspieces-train", "chessvision-classification"),
        #     "val": tlc.Url.create_table_url("val-cleaned-filtered", "chesspieces-val", "chessvision-classification"),
        # },
        # settings=settings,
        data="data/board_extraction/yolo/data.yaml",
        batch=2,
        imgsz=256,
        epochs=10,
        workers=4,
        project=str(Path(__file__).parent.parent / "yolo_output"),
    )

    print("Running tests with trained model...")
    # del model
    # model = YOLOModelWrapper(TLCYOLO(results.save_dir / "weights" / "best.pt"))
    # run_tests(run=tlc.active_run(), classifier=model)

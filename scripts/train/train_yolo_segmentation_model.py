from pathlib import Path

import tlc
from ultralytics.utils.tlc import TLCYOLO, Settings

# First time, use this data
# data="data/board_extraction/yolo/data.yaml",

if __name__ == "__main__":
    model = TLCYOLO("yolo11s-seg.pt")

    settings = Settings(
        project_name="chessvision-yolo-segmentation",
        run_description="Use small model",
        image_embeddings_dim=2,
        conf_thres=0.2,
        sampling_weights=True,
        exclude_zero_weight_training=True,
        exclude_zero_weight_collection=False,
    )
    results = model.train(
        tables={
            "train": tlc.Url.create_table_url("initial", "train", "chessvision-yolo-segmentation"),
            "val": tlc.Url.create_table_url("initial", "val", "chessvision-yolo-segmentation"),
        },
        settings=settings,
        batch=4,
        imgsz=256,
        epochs=10,
        workers=4,
        project=str(Path(__file__).parent.parent.parent / "yolo_output"),
    )

    print("Running tests with trained model...")
    model_path = results.save_dir / "weights" / "best.pt"
    from scripts.eval import evaluate_model

    evaluate_model(
        board_extractor_weights=model_path,
        board_extractor_model_id="yolo",
        classifier_model_id="yolo",
        include_metrics_table=True,
        table_name="merged-2024-11-04-2024-11-04",
    )

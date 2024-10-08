import tlc
import torch
from ultralytics.utils.tlc import TLCYOLO, Settings


class YOLOModelWrapper:
    def __init__(self, model: TLCYOLO):
        self.model = model

    def __call__(self, img):
        res = self.model(img.repeat(1, 3, 1, 1))
        return torch.vstack([r.probs.data for r in res])


if __name__ == "__main__":
    model = TLCYOLO("yolov8m-cls.pt")

    settings = Settings(
        image_embeddings_dim=2,
        conf_thres=0.2,
        project_name="chessvision-classification",
        run_description="Train on initial training set",
        sampling_weights=True,
        exclude_zero_weight_training=True,
        exclude_zero_weight_collection=False,
    )
    results = model.train(
        tables={
            "train": "C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/chessvision-classification/datasets/chesspieces-train/tables/train",
            "val": "C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/chessvision-classification/datasets/chesspieces-val/tables/val-cleaned-filtered",
        },
        settings=settings,
        batch=-1,
        imgsz=64,
        epochs=10,
        workers=4,
    )

    from chessvision.test import run_tests

    print("Running tests...")
    run = tlc.active_run()
    del model
    model = YOLOModelWrapper(TLCYOLO(results.save_dir / "weights" / "best.pt"))

    run_tests(run=run, classifier=model)

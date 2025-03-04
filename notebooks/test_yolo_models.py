import torch
from ultralytics import YOLO

from chessvision.test import run_tests


class CLSYOLOModelWrapper:
    def __init__(self, model: YOLO):
        self.model = model

    def __call__(self, img):
        res = self.model(img.repeat(1, 3, 1, 1))
        return torch.vstack([r.probs.data for r in res])


class SEGYOLOModelWrapper:
    def __init__(self, model: YOLO):
        self.model = model

    def __call__(self, img):
        res = self.model(img)
        try:
            probs = [r.masks.data for r in res]
        except AttributeError:
            return torch.zeros((len(res), 1, 256, 256))
        return torch.cat(probs)


seg_path = "yolo_output/train4/weights/best.pt"
cls_path = "C:/Project/ultralytics-3lc/runs/classify/train33/weights/best.pt"

classifier = CLSYOLOModelWrapper(YOLO(cls_path))
segmentation_model = SEGYOLOModelWrapper(YOLO(seg_path))

run_tests(classifier=classifier, extractor=segmentation_model)

from __future__ import annotations

from typing import Any

import tlc
import torch
import torch.nn as nn

from chessvision import utils
from chessvision.pytorch_unet.utils.dice_score import dice_loss
from chessvision.utils import get_device


class LossCollector(tlc.MetricsCollector):
    def __init__(self) -> None:
        super().__init__()
        self.device = utils.get_device()

    def compute_metrics(
        self, batch: dict[str, Any], predictor_output: tlc.PredictorOutput
    ) -> dict[str, Any]:
        predictions = predictor_output.forward
        _, masks = batch["image"], batch["mask"]
        masks = masks.to(self.device)

        use_channels_last = self.device.type in ["cuda", "cpu"]
        if use_channels_last:
            predictions = predictions.to(memory_format=torch.channels_last)

        unreduced_criterion = nn.BCEWithLogitsLoss(reduction="none")

        unreduced_dice_loss = dice_loss(
            torch.sigmoid(predictions),
            masks.float(),
            multiclass=False,
            reduce_batch_first=False,
            reduction="none",
        )

        unreduced_bce_loss = unreduced_criterion(predictions, masks.float()).mean(
            (-1, -2)
        )

        loss = unreduced_dice_loss + unreduced_bce_loss

        return {
            "loss": loss.cpu().numpy().squeeze(),
        }

    @property
    def column_schemas(self) -> dict[str, tlc.Schema]:
        return {}

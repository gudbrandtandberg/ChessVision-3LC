from __future__ import annotations

from typing import Any

import tlc
import torch
import torch.nn as nn

from chessvision.core import ChessVision
from chessvision.pytorch_unet.utils.dice_score import dice_loss


class LossCollector(tlc.MetricsCollector):
    def __init__(self):
        super().__init__()
        self.device = ChessVision.get_device()

    def compute_metrics(self, batch, predictor_output) -> dict[str, Any]:
        predictions = predictor_output.forward
        _, masks = batch["image"], batch["mask"]
        masks = masks.to(self.device)

        unreduced_criterion = nn.BCEWithLogitsLoss(reduction="none")

        unreduced_dice_loss = dice_loss(
            torch.sigmoid(predictions), masks.float(), multiclass=False, reduce_batch_first=False, reduction="none"
        )

        unreduced_bce_loss = unreduced_criterion(predictions, masks.float()).mean((-1, -2))

        loss = unreduced_dice_loss + unreduced_bce_loss

        metrics_batch = {
            "loss": loss.cpu().numpy().squeeze(),
        }
        return metrics_batch

    @property
    def column_schemas(self) -> dict[str, tlc.Schema]:
        return {}

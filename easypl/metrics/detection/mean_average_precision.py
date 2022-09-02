from typing import Optional, List, Dict

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import Metric


class MAP(Metric):
    def __init__(
            self,
            iou_thresholds: Optional[List[float]] = None,
            rec_thresholds: Optional[List[float]] = None,
            max_detection_thresholds: Optional[List[int]] = None,
            class_metrics: bool = False,
            **kwargs
    ):
        super().__init__()
        self.metric = MeanAveragePrecision(
            box_format='xyxy',
            iou_type='bbox',
            iou_thresholds=iou_thresholds,
            rec_thresholds=rec_thresholds,
            max_detection_thresholds=max_detection_thresholds,
            class_metrics=class_metrics,
            **kwargs
        )

    def update(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor
    ):
        preds_ = [
            dict(
                boxes=pred[:, :4],
                scores=pred[:, 4],
                labels=pred[:, 5].long(),
            ) for pred in preds]

        targets_ = [
            dict(
                boxes=target[torch.where(target[:, 4] > -1)[0]][:, :4],
                labels=target[torch.where(target[:, 4] > -1)[0]][:, 4]
            ) for target in targets
        ]
        self.metric.update(preds_, targets_)

    def compute(
            self
    ) -> Dict:
        return self.metric.compute()

    def reset(self):
        self.metric.reset()

from typing import Optional, List, Dict

import torch
from torchmetrics import Metric


class MAP(Metric):
    """
    Wrapper for MeanAveragePrecision from torchmetrics.

    Attributes
    ----------------
    iou_thresholds: Optional[List[float]]
        Iou threshold/thresholds for boxes.

    rec_thresholds: Optional[List[float]]
        Recall thresholds for evaluation.

    max_detection_thresholds: Optional[List[int]]
        Thresholds on max detections per image.

    class_metrics: bool
        Option to enable per-class metrics for mAP and mAR_100.

    kwargs
        Torchmetrics Metric args.

    """

    def __init__(
            self,
            iou_thresholds: Optional[List[float]] = None,
            rec_thresholds: Optional[List[float]] = None,
            max_detection_thresholds: Optional[List[int]] = None,
            class_metrics: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
        except ImportError:
            raise ImportError('MeanAveragePrecision doesn\'t available in your torchmetrics version.')
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
                boxes=pred[:, :4].cpu(),
                scores=pred[:, 4].cpu(),
                labels=pred[:, 5].long().cpu(),
            ) for pred in preds]

        targets_ = [
            dict(
                boxes=target[torch.where(target[:, 4] > -1)[0]][:, :4].cpu(),
                labels=target[torch.where(target[:, 4] > -1)[0]][:, 4].cpu()
            ) for target in targets
        ]
        self.metric.update(preds_, targets_)

    def compute(
            self
    ) -> Dict:
        return self.metric.compute()

    def reset(self):
        self.metric.reset()

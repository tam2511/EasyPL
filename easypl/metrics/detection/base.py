from typing import Union, List, Optional, Dict

import torch
from torchvision.ops.boxes import box_iou
from torchmetrics import Metric


class BaseDetectionMetric(Metric):
    """
    Base detection metric. Compute true positive, false negative and false positive metrics.

    Attributes
    ----------------
    iou_threshold: Union[float, List[float]]
        Iou threshold/thresholds for boxes.

    confidence: Union[float, List[float]]
        Confidence/confidences thresholds.

    num_classes: Optional[int]
        Number of classes.

    kwargs
        Torchmetrics Metric args.

    """
    def __init__(
            self,
            iou_threshold: Union[float, List[float]],
            confidence: Union[float, List[float]],
            num_classes: Optional[int] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        iou_thresholds = torch.tensor(
            iou_threshold if isinstance(iou_threshold, list) else [iou_threshold]
        )
        confidences = torch.tensor(
            confidence if isinstance(confidence, list) else [confidence]
        ).sort().values
        self.num_classes = num_classes

        buffers_size = (confidences.size(0), iou_thresholds.size(0),) if self.num_classes is None else (
            num_classes, confidences.size(0), iou_thresholds.size(0)
        )

        self.add_state("iou_thresholds", default=iou_thresholds, dist_reduce_fx="sum")
        self.add_state("confidences", default=confidences, dist_reduce_fx="sum")
        self.add_state("tp", default=torch.zeros(*buffers_size), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(*buffers_size), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(*buffers_size), dist_reduce_fx="sum")

    def update(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor
    ):
        preds_boxes = preds[:, :, :4]
        targets_boxes = targets[:, :, :4]

        preds_classes = preds[:, :, 5]
        targets_classes = targets[:, :, 4]

        preds_probs = preds[:, :, 4]

        # TODO: override with batch ops
        for idx in range(len(preds)):
            filtred_preds_idxs = torch.where(preds_probs[idx] > self.confidences[0])[0]
            filtred_preds_boxes = preds_boxes[idx][filtred_preds_idxs]
            filtred_preds_classes = preds_classes[idx][filtred_preds_idxs]
            filtred_preds_probs = preds_probs[idx][filtred_preds_idxs]

            filtred_targets_idxs = torch.where(targets_classes[idx] > 0)[0]
            filtred_targets_boxes = targets_boxes[idx][filtred_targets_idxs]
            filtred_targets_classes = targets_classes[idx][filtred_targets_idxs]
            unique_classes = filtred_targets_classes.unique(return_counts=True)

            # [N x M]
            boxes_ious = box_iou(filtred_preds_boxes, filtred_targets_boxes)

            # [THRESHOLDS x N x M]
            boxes_ious = boxes_ious.unsqueeze(0).repeat(
                len(self.iou_thresholds), 1, 1
            ) > self.iou_thresholds.unsqueeze(-1).unsqueeze(-1).repeat(
                1, *boxes_ious.shape
            )

            # [N x M]
            classes_confus_matrix = filtred_preds_classes.unsqueeze(-1).repeat(
                1, len(filtred_targets_classes)
            ) == filtred_targets_classes.unsqueeze(0).repeat(
                len(filtred_preds_classes), 1
            )

            # [THRESHOLDS x N x M]
            pred_matrix = boxes_ious * classes_confus_matrix.unsqueeze(0).repeat(
                len(self.iou_thresholds), 1, 1
            )

            for confidence_idx in range(len(self.confidences)):
                filtred_preds_idxs = torch.where(filtred_preds_probs > self.confidences[confidence_idx])[0]

                if len(filtred_preds_idxs) == 0:
                    if self.num_classes is None:
                        self.fn[confidence_idx, :] += len(filtred_targets_idxs)
                    else:
                        self.fn[unique_classes[0], confidence_idx, :] += unique_classes[1].unsqueeze(-1).repeat(
                            1, len(self.iou_thresholds)
                        )
                    continue

                # [THRESHOLDS x M]
                results = pred_matrix[:, filtred_preds_idxs, :].sum(dim=1)
                tp_ = results > 0
                fp_ = torch.where(results > 0, results - 1, results)
                fn_ = results == 0
                if self.num_classes is None:
                    self.tp[confidence_idx, :] += tp_.sum(dim=1)
                    self.fp[confidence_idx, :] += fp_.sum(dim=1)
                    self.fn[confidence_idx, :] += fn_.sum(dim=1)
                else:
                    # TODO: override with batch ops
                    for class_idx in range(len(filtred_targets_classes)):
                        self.tp[filtred_targets_classes[class_idx], confidence_idx, :] += tp_[:, class_idx]
                        self.fp[filtred_targets_classes[class_idx], confidence_idx, :] += fp_[:, class_idx]
                        self.fn[filtred_targets_classes[class_idx], confidence_idx, :] += fn_[:, class_idx]

    def compute(
            self
    ) -> Dict:
        return {
            'tp': self.tp,
            'fn': self.fn,
            'fp': self.fp,
        }

    def reset(self):
        self.tp *= 0
        self.fn *= 0
        self.fp *= 0

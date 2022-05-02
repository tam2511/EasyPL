from typing import Optional, List

from torchmetrics import *
import torch


class TorchMetric(Metric):
    """
    Wrapper for metrics from torchmetrics

    Attributes
    -----------
    metric: Metric
        Metric object from torchmetrics.

    class_names: Optional[List]
        Names of classes.

    Examples
    ------------
    >>> from torchmetrics import F1
    ... from easypl.metrics import TorchMetric
    ... result_metric = TorchMetric(F1(), class_names=None)

    """

    def __init__(
            self,
            metric: Metric,
            class_names: Optional[List] = None
    ):
        super().__init__(dist_sync_on_step=metric.dist_sync_on_step, compute_on_step=metric.compute_on_step)
        self.class_names = class_names
        self.metric = metric
        self.name = metric.__class__.__name__

    def compute(self) -> dict:
        result = self.metric.compute()
        if not isinstance(result, torch.Tensor):
            raise ValueError(
                f'TorchMetric does not support {self.name}. Result type is {type(result)}, not torch.Tensor.')
        if result.ndim > 1:
            raise ValueError(
                f'TorchMetric does not support {self.name}. Number dims of result: {result.ndim} > 1.')
        if result.ndim == 0:
            return {'{}_{}'.format(self.name, self.metric.average): result}
        classes = torch.arange(result.size(0)) if self.class_names is None else self.class_names
        if result.size(0) != len(classes):
            raise ValueError(
                f'Length of result {self.name} {result.size(0)} not equal number of classes {len(classes)}')
        return {'{}_{}'.format(self.name, classes[idx]): result[idx] for idx in range(len(classes))}

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.metric.update(preds, targets)

    def reset(self):
        self.metric.reset()

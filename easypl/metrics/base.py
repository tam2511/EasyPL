from torchmetrics import Metric
from torch.nn import ModuleList
import torch


class MetricsList(Metric):
    """
    List of metrics
    """

    def __init__(
            self,
            dist_sync_on_step: bool = False,
            compute_on_step: bool = True
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.metrics = ModuleList()
        self.cache_metrics = ModuleList()
        if not hasattr(self, '_device'):
            self._device = torch.device('cpu')

    def __to(self, device: torch.device):
        for metric_idx in range(len(self.metrics)):
            self.metrics[metric_idx].to(device=device)
        self.to(device)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for metric_idx in range(len(self.metrics)):
            if self._device != preds.device:
                self.__to(preds.device)
            self.metrics[metric_idx].update(preds, target)

    def compute(self):
        result = {}
        for metric_idx in range(len(self.metrics)):
            result_ = self.metrics[metric_idx].compute()
            result.update(result_)
        return result

    def reset(self):
        for metric_idx in range(len(self.metrics)):
            self.metrics[metric_idx].reset()

    def add(self, metric: Metric):
        self.metrics.append(module=metric)
        self.cache_metrics.append(module=metric)

    def clone(self):
        metric_list = self.__class__()
        for metric in self.cache_metrics:
            metric_list.add(metric)
        return metric_list

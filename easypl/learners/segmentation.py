from typing import Optional, Union, List
import torch
from torchmetrics import Metric

from easypl.learners.base import BaseLearner
from easypl.optimizers import WrapperOptimizer
from easypl.lr_schedulers import WrapperScheduler


class SegmentationLearner(BaseLearner):
    def __init__(
            self,
            model: Optional[Union[torch.nn.Module, List[torch.nn.Module]]] = None,
            loss: Optional[Union[torch.nn.Module, List[torch.nn.Module]]] = None,
            optimizer: Optional[Union[WrapperOptimizer, List[WrapperOptimizer]]] = None,
            lr_scheduler: Optional[Union[WrapperScheduler, List[WrapperScheduler]]] = None,
            train_metrics: Optional[List[Metric]] = None,
            val_metrics: Optional[List[Metric]] = None,
            test_metrics: Optional[List[Metric]] = None,
            data_keys: Optional[List[str]] = None,
            target_keys: Optional[List[str]] = None,
            multilabel: bool = False

    ):
        super().__init__(
            model=model,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            data_keys=data_keys,
            target_keys=target_keys,
        )
        if len(data_keys) != 1 and len(target_keys) != 1:
            raise ValueError('"data_keys" and "target_keys" must be one element')
        self.multilabel = multilabel

    __init__.__doc__ = BaseLearner.__init__.__doc__

    def forward(self, samples):
        return self.model(samples)

    def loss_step(self, outputs, targets):
        loss = self.loss_f(
            outputs,
            targets.float() if self.multilabel or targets.ndim != 1 else targets.long()
        )
        return {
            'loss': loss,
            'log': {
                'loss': loss
            }
        }

    def get_targets(self, batch):
        targets = batch[self.target_keys[0]]
        return {
            'loss': targets.float() if self.multilabel or targets.ndim > 3 else targets.long(),
            'metric': targets if targets.ndim == 3 or self.multilabel else targets.argmax(dim=1),
            'log': targets
        }

    def get_outputs(self, batch):
        samples = batch[self.data_keys[0]]
        outputs = self.forward(samples)
        return {
            'loss': outputs,
            'metric': outputs.sigmoid() if self.multilabel else outputs.argmax(dim=1),
            'log': outputs.sigmoid() if self.multilabel else outputs.softmax(dim=1),
        }

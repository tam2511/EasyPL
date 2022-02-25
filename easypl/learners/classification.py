from typing import Optional, Union, List
import torch
from torchmetrics import Metric

from easypl.learners.base import BaseLearner
from easypl.optimizers import WrapperOptimizer
from easypl.lr_schedulers import WrapperScheduler


class ClassificatorLearner(BaseLearner):
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
            target_keys=target_keys
        )
        if len(data_keys) != 1 and len(target_keys) != 1:
            raise ValueError('"data_keys" and "target_keys" must be one element')
        self.multilabel = False

    __init__.__doc__ = BaseLearner.__init__.__doc__

    def forward(self, inputs):
        return self.model(inputs)

    def common_step(self, batch, batch_idx):
        images = batch[self.data_keys[0]]
        targets = batch[self.target_keys[0]]
        self.multilabel = targets.ndim > 1
        output = self.forward(images)
        loss = self.loss_f(output, targets.float() if self.multilabel else targets)
        return {
            'loss': loss,
            'output_for_metric': output.sigmoid() if self.multilabel else output.argmax(dim=1),
            'target_for_metric': targets,
            'output_for_log': output.sigmoid() if self.multilabel else output.softmax(dim=1),
            'target_for_log': targets
        }

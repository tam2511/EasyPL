from typing import Optional

import torch
from pytorch_lightning import Callback, Trainer

from easypl.learners import BaseLearner
from easypl.optimizers import WrapperOptimizer


class OptimizerInitialization(Callback):
    def optimizer_initialization(
            self,
            model: torch.nn.Module,
            optimizer_wrapper: WrapperOptimizer,
            optimizer_idx: int = 0
    ):
        """
        Callback for optimizer initialization on fit start

        Attributes
        ----------
        model: torch.nn.Module
            Pytorch model object.

        optimizer_wrapper: WrapperOptimizer
            Object of WrapperOptimizer. See: https://easypl.readthedocs.io/en/latest/apis/Optimizers.html.

        optimizer_idx: int
            Index of optimizer.

        Examples
        ----------
            >>> from easypl.callbacks import OptimizerInitialization
            ...
            ...
            ... class YoloV5OptimizerInitialization(OptimizerInitialization):
            ...    def __init__(self, decay=1e-5):
            ...        super().__init__()
            ...        self.decay = decay
            ...
            ...    def optimizer_initialization(self, model, optimizer_wrapper, optimizer_idx=0):
            ...        g = [], [], []  # optimizer parameter groups
            ...        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
            ...        for v in model.modules():
            ...            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            ...                g[2].append(v.bias)
            ...            if isinstance(v, bn):  # weight (no decay)
            ...                g[1].append(v.weight)
            ...            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            ...                g[0].append(v.weight)
            ...
            ...        optimizer = optimizer_wrapper(g[2])
            ...
            ...        optimizer.add_param_group({'params': g[0], 'weight_decay': self.decay})  # add g0 with weight_decay
            ...        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
            ...        return optimizer

        """
        raise NotImplementedError

    def setup(
            self,
            trainer: Trainer,
            pl_module: BaseLearner,
            stage: Optional[str] = None
    ) -> None:
        if pl_module.precomputed_optimizer is not None or pl_module.optimizer is None:
            return
        if isinstance(pl_module.optimizer, list):
            if not isinstance(pl_module.model, torch.nn.ModuleList):
                raise ValueError('For multiple optimizers need multiple models with same len.')
            elif len(pl_module.model) != len(pl_module.optimizer):
                raise ValueError('Number of models must be equal number of optimizers')
            else:
                optimizers = [
                    self.optimizer_initialization(pl_module.model[idx], pl_module.optimizer[idx], idx)
                    for idx in range(len(pl_module.model))
                ]
        else:
            optimizers = [
                self.optimizer_initialization(pl_module.model, pl_module.optimizer)
            ]
        pl_module.precomputed_optimizer = optimizers

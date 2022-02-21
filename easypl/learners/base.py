from typing import Optional, Union, List
from numbers import Number
import warnings

from pytorch_lightning import LightningModule
import torch
from torchmetrics import Metric

from easypl.lr_schedulers import WrapperScheduler
from easypl.metrics import MetricsList
from easypl.optimizers import WrapperOptimizer


class BaseLearner(LightningModule):
    def __init__(
            self,
            model: Optional[Union[torch.nn.Module, List[torch.nn.Module]]] = None,
            loss: Optional[Union[torch.nn.Module, List[torch.nn.Module]]] = None,
            optimizer: Optional[Union[WrapperOptimizer, List[WrapperOptimizer]]] = None,
            lr_scheduler: Optional[Union[WrapperScheduler, List[WrapperScheduler]]] = None,
            train_metrics: Optional[List[Metric]] = None,
            val_metrics: Optional[List[Metric]] = None,
            data_keys: Optional[List[str]] = None,
            target_keys: Optional[List[str]] = None
    ):
        """
        :param model: torch.nn.Module model
        :param loss: torch.nn.Module loss function
        :param optimizer: Optimizer wrapper object
        :param lr_scheduler: Scheduler object for lr scheduling
        :param train_metrics: list of train metrics
        :param val_metrics:list of val metrics
        """
        super().__init__()
        self.model = model
        self.loss_f = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_metrics = MetricsList()
        if not train_metrics is None:
            for train_metric in train_metrics:
                self.train_metrics.add(metric=train_metric)
        self.val_metrics = MetricsList([])
        if not val_metrics is None:
            for val_metric in val_metrics:
                self.val_metrics.add(metric=val_metric)
        if self.data_keys is None or self.target_keys is None:
            raise ValueError('"data_keys" and "target_keys" can not be None')
        self.data_keys = data_keys
        self.target_keys = target_keys
        self.return_train_output = True
        self.return_val_output = True

    def common_step(self, batch, batch_idx) -> dict:
        """
        @return: {
            'loss': tensor or dict with 'main' key,
            'output_for_metric': ...,
            'target_for_metric': ...,
            'output_for_log': ...,
            'target_for_log': ...
        }
        """
        raise NotImplementedError

    def __log(self, name: str, obj, on_step: bool = True, on_epoch: bool = False):
        if isinstance(obj, dict):
            for key in obj:
                self.__log('_'.join([name, key]), obj[key], on_step=on_step, on_epoch=on_epoch)
        elif isinstance(obj, torch.Tensor) or isinstance(obj, Number):
            self.log(name, obj, on_step=on_step, on_epoch=on_epoch)
        else:
            warnings.warn(f'Value with name {name} has unsupported type {type(obj)}. This value can\'t logged.',
                          Warning, stacklevel=2)

    def __log_lr_optimizer(self, optimizer: torch.optim.Optimizer, optimizer_idx=None):
        optimizer_name = 'optimizer' if optimizer_idx is None else f'optimizer_{optimizer_idx}'
        lrs = [group['lr'] for group in optimizer.param_groups]
        grouped_lrs = {}
        for idx, lr in enumerate(lrs):
            if lr not in grouped_lrs:
                grouped_lrs[lr] = []
            grouped_lrs[lr].append(idx)
        if len(grouped_lrs) == 1:
            self.log(f'{optimizer_name}/lr', lrs[0], on_step=True, on_epoch=False, prog_bar=True)
        else:
            for lr in grouped_lrs:
                ids = ','.join(map(str, grouped_lrs[lr]))
                self.log(f'{optimizer_name}/lr_groups[{ids}]', lr, on_step=True, on_epoch=False, prog_bar=True)

    def __log_lr(self):
        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            for idx, optimizer in enumerate(optimizers):
                self.__log_lr_optimizer(optimizer=optimizer, optimizer_idx=idx)
        else:
            self.__log_lr_optimizer(optimizer=optimizers)

    def training_step(self, batch, batch_idx):
        result = self.common_step(batch, batch_idx)
        self.__log('train/loss', result['loss'], on_step=True, on_epoch=False)
        self.__log_lr()
        self.train_metrics.update(result['output_for_metric'], result['target_for_metric'])
        ret = {'loss': result['loss'] if isinstance(torch.Tensor) else result['loss']['main']}
        if self.return_train_output:
            ret['output'] = result['output_for_log']
            ret['target'] = result['target_for_log']
        return ret

    def training_epoch_end(self, train_step_outputs):
        train_metrics = self.train_metrics.compute()
        self.train_metrics.reset()
        for metric_name in train_metrics:
            self.log(f'train/{metric_name}', train_metrics[metric_name], on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        result = self.common_step(batch, batch_idx)
        self.__log('val/loss', result['loss'], on_step=True, on_epoch=False)
        self.val_metrics.update(result['output_for_metric'], result['target_for_metric'])
        ret = {'loss': result['loss'] if isinstance(torch.Tensor) else result['loss']['main']}
        if self.return_val_output:
            ret['output'] = result['output_for_log']
            ret['target'] = result['target_for_log']
        return ret

    def validation_epoch_end(self, val_step_outputs):
        val_metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        for metric_name in val_metrics:
            self.log(f'val/{metric_name}', val_metrics[metric_name], on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if isinstance(self.optimizer, list):
            if not isinstance(self.model, list):
                raise ValueError('For multiple optimizers need multiple models with same len.')
            elif len(self.model) != len(self.optimizer):
                raise ValueError('Number of models must be equal number of optimizers')
            else:
                optimizers = [
                    self.optimizer[idx](filter(lambda p: p.requires_grad, self.model[idx].parameters()))
                    for idx in range(len(self.model))
                ]
        else:
            if isinstance(self.model, list):
                optimizers = [
                    self.optimizer(filter(lambda p: p.requires_grad, self.model[idx].parameters()))
                    for idx in range(len(self.model))
                ]
            else:
                optimizers = [
                    self.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()))
                ]
        if isinstance(self.lr_scheduler, list):
            if len(optimizers) != len(self.lr_scheduler):
                raise ValueError('Number of lr_schedulers must be equal number of optimizers')
            else:
                lr_schedulers = [
                    self.lr_scheduler[idx](optimizers[idx]) for idx in range(len(optimizers))
                ]
        else:
            lr_schedulers = [
                self.lr_scheduler(optimizers[idx]) for idx in range(len(optimizers))
            ]
        return optimizers, lr_schedulers

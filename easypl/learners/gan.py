from typing import Optional, Union, List, Any, Dict
import torch
from torchmetrics import Metric

from easypl.learners.base import BaseLearner
from easypl.optimizers import WrapperOptimizer
from easypl.lr_schedulers import WrapperScheduler


class GANLearner(BaseLearner):
    """
    Generative adversarial networks learner.

    Attributes
    ----------
    model: Optional[List[torch.nn.Module]]
        Generative adversarial networks.

    loss: Optional[List[torch.nn.Module]]
        torch.nn.Module losses function.

    optimizer: Optional[List[WrapperOptimizer]]
        Optimizers wrapper object.

    lr_scheduler: Optional[Union[WrapperScheduler, List[WrapperScheduler]]]
        Scheduler object for lr scheduling.

    train_metrics: Optional[List[Metric]]
        List of train metrics.

    val_metrics: Optional[List[Metric]]
        List of validation metrics.

    test_metrics: Optional[List[Metric]]
        List of test metrics.

    data_keys: Optional[List[str]]
        List of data keys

    target_keys: Optional[List[str]]
        List of target keys

    """
    def __init__(
            self,
            model: Optional[List[torch.nn.Module]] = None,
            loss: Optional[List[torch.nn.Module]] = None,
            optimizer: Optional[List[WrapperOptimizer]] = None,
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
        if len(data_keys) != 1:
            raise ValueError('"data_keys" and "target_keys" must be one element')

    __init__.__doc__ = BaseLearner.__init__.__doc__

    def forward(
            self,
            samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Standart method for forwarding model.
        Attributes
        ----------
        samples: torch.Tensor
            Image tensor.

        Returns
        ----------
        torch.Tensor
            Output from model.
        """
        return self.model(samples)

    def loss_step(
            self,
            outputs: Dict,
            targets: Dict,
            optimizer_idx: int = 0
    ) -> Dict:
        """
        Method fow loss evaluating.

        Attributes
        ----------
        outputs: torch.Tensor
            Outputs from model

        targets: torch.Tensor
            Targets from batch

        optimizer_idx: int
            Index of optimizer

        Returns
        ----------
        Dict
            Dict with keys: ["loss", "log"]
        """
        if optimizer_idx == 0:
            loss = self.loss_f(
                outputs['fake'],
                targets['valid']
            )
            return {
                'loss': loss,
                'log': {
                    'g_loss': loss
                }
            }
        if optimizer_idx == 1:
            real_loss = self.loss_f(
                outputs['valid'], targets['valid']
            )
            fake_loss = self.loss_f(
                outputs['fake'], targets['fake']
            )
            loss = (real_loss + fake_loss) / 2
            return {
                'loss': loss,
                'log': {
                    'real_loss': real_loss,
                    'fake_loss': fake_loss,
                }
            }

    def get_targets(
            self,
            batch: Dict,
            optimizer_idx: int = 0
    ) -> Dict:
        """
        Method for selecting and preprocessing targets from batch

        Attributes
        ----------
        batch: Dict
            Batch in step

        optimizer_idx: int
            Index of optimizer

        Returns
        ----------
        Dict
            Dict with keys: ["loss", "metric", "log"]
        """
        targets = batch[self.target_keys[0]]
        valid = torch.ones(targets.size(0), 1)
        valid = valid.type_as(targets)
        fake = torch.zeros(targets.size(0), 1)
        fake = fake.type_as(targets)
        return {
            'loss': {
                'fake': fake,
                'valid': valid
            },
            'metric': targets,
            'log': targets
        }

    def get_outputs(
            self,
            batch: Dict,
            optimizer_idx: int = 0
    ) -> Dict:
        """
        Abtract method for selecting and preprocessing outputs from batch

        Attributes
        ----------
        batch: Dict
            Batch in step

        optimizer_idx: int
            Index of optimizer

        Returns
        ----------
        Dict
            Dict with keys: ["loss", "metric", "log"]
        """
        samples = batch[self.data_keys[0]]
        outputs = self.forward(samples)
        fake_disc = self.model[1](outputs)
        if optimizer_idx == 0:
            return {
                'loss': {
                    'fake': fake_disc
                },
                'metric': outputs,
                'log': outputs,
            }

        if optimizer_idx == 1:
            targets = batch[self.target_keys[0]]
            return {
                'loss': {
                    'fake': fake_disc,
                    'valid': self.model[1](targets)
                },
                'metric': None,
                'log': None,
            }

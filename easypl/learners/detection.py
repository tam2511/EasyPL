from typing import Optional, Union, List, Any, Dict
import torch
from torchmetrics import Metric

from easypl.learners.base import BaseLearner
from easypl.optimizers import WrapperOptimizer
from easypl.lr_schedulers import WrapperScheduler
from easypl.utilities.detection import BasePostprocessing


class DetectionLearner(BaseLearner):
    """
    Detection learner.

    Attributes
    ----------
    model: Optional[Union[torch.nn.Module, List[torch.nn.Module]]]
        torch.nn.Module model.

    loss: Optional[Union[torch.nn.Module, List[torch.nn.Module]]]
        torch.nn.Module loss function.

    optimizer: Optional[Union[WrapperOptimizer, List[WrapperOptimizer]]]
        Optimizer wrapper object.

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

    postprocessing: Optional
        If postprocessing is not None then this
    """
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
            image_size_key: Optional[str] = None,
            image_scale_key: Optional[str] = None,
            postprocessing: Optional[BasePostprocessing] = None
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
        self.image_size_key = image_size_key
        self.image_scale_key = image_scale_key
        self.postprocessing = postprocessing

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
            outputs: torch.Tensor,
            targets: torch.Tensor,
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
        losses = self.loss_f(
            outputs,
            targets
        )
        return {
            'loss': losses['loss'],
            'log': losses
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
        image_scales = batch[self.image_scale_key]
        transformed_targets = targets if self.postprocessing is None else self.postprocessing.targets_handle(
            targets,
            image_scales
        )
        return {
            'loss': targets,
            'metric': transformed_targets,
            'log': transformed_targets
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
        image_sizes = batch[self.image_size_key]
        image_scales = batch[self.image_scale_key]
        transformed_outputs = outputs if self.postprocessing is None else self.postprocessing.outputs_handle(
            outputs,
            image_sizes,
            torch.ones_like(image_scales, dtype=image_scales.dtype, device=image_scales.device),
        )
        return {
            'loss': outputs,
            'metric': transformed_outputs,
            'log': transformed_outputs,
        }

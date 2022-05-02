from typing import List, Dict, Union, Callable, Any, Tuple

import pytorch_lightning
import torch

from easypl.callbacks.predictors.base_image import BaseImageTestTimeAugmentation


class ClassificationImageTestTimeAugmentation(BaseImageTestTimeAugmentation):
    """
        Image classification callback for test-time-augmentation

        Attributes
        ----------
        n: int
            Number of augmentations.

        augmentations: List
            List of augmentations, which will be used.

        augmentation_method: str
            Method of selecting augmentations from list. Available: ["first", "random"]

        phase: str
            Phase which will be used by this predictor callback.
            Available: ["val", "test", "predict"].

        reduce_method: Union[str, Callable]
            Method of result reducing

        """

    def __init__(
            self,
            n: int,
            augmentations: List,
            augmentation_method: str = 'first',
            phase='val',
            reduce_method: Union[str, Callable] = 'mean'
    ):
        super().__init__(n=n, augmentations=augmentations, augmentation_method=augmentation_method, phase=phase)
        self.reduce_method = reduce_method
        self.multilabel = None

    def post_init(
            self,
            trainer: pytorch_lightning.Trainer,
            pl_module: pytorch_lightning.LightningModule
    ):
        """
        Method for initialization in first batch handling. [NOT REQUIRED]

        Attributes
        ----------
        trainer: pytorch_lightning.Trainer
            Trainer of pytorch-lightning

        pl_module: pytorch_lightning.LightningModule
            LightningModule of pytorch-lightning

        """
        super().post_init(trainer, pl_module)
        self.multilabel = pl_module.multilabel
        if len(self.data_keys) != 1:
            raise ValueError(f'Size of "data_keys" must have len 1, but have len {len(self.data_keys)}')

    def metric_formatting(
            self,
            outputs: Any,
            targets: Any
    ) -> Tuple:
        """
        Preparing before metric pass.

        Attributes
        ----------
        outputs: Any
            Output from model

        targets: Any
            Targets from batch

        Returns
        ----------
        Tuple
            Formatted outputs and targets
        """
        targets = targets if targets.ndim == 1 or self.multilabel else targets.argmax(dim=1)
        return outputs, targets

    def reduce(
            self,
            tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Method for reducing of results.

        Attributes
        ----------
        tensor: torch.Tensor
            Any tensor with size [batch_size X ...]

        Returns
        ----------
        torch.Tensor
            Reduced tensor

        """
        if isinstance(self.reduce_method, str):
            if self.reduce_method == 'mean':
                return tensor.mean(0)
            else:
                raise NotImplementedError
        else:
            return self.reduce_method(tensor)

    def augment(
            self,
            sample: Dict,
            augmentation
    ) -> Dict:
        """
        Method for augmentation apply.

        Attributes
        ----------
        sample: Dict
            Any sample of batch

        augmentation
            Transform object

        Returns
        ----------
        Dict
            Augmented sample
        """
        image = sample[self.data_keys[0]].copy()
        image = augmentation(image=image)['image']
        ret_sample = {key: sample[key] for key in sample if key not in self.data_keys}
        ret_sample[self.data_keys[0]] = image
        return ret_sample

    def preprocessing(
            self,
            sample: Dict,
            dataloader_idx: int = 0
    ) -> Dict:
        """
        Method for preprocessing sample

        Attributes
        ----------
        sample: Dict
            Any sample of batch

        dataloader_idx: int
            Index of dataloader

        Returns
        ----------
        Dict
            Preprocessed sample
        """
        sample[self.data_keys[0]] = self.inv_transform[dataloader_idx](
            image=sample[self.data_keys[0]].cpu().numpy()
        )['image']
        return sample

    def postprocessing(
            self,
            sample: Dict,
            dataloader_idx: int = 0
    ) -> Dict:
        """
        Method for postprocessing sample

        Attributes
        ----------
        sample: Dict
            Any sample of batch

        dataloader_idx: int
            Index of dataloader

        Returns
        ----------
        Dict
            Postprocessed sample
        """
        sample[self.data_keys[0]] = self.transform[dataloader_idx](image=sample[self.data_keys[0]])['image']
        return sample

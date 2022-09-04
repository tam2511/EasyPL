from typing import List, Dict

import pytorch_lightning

from easypl.callbacks.predictors.base import BaseTestTimeAugmentation
from easypl.utilities.transforms import inv_transform, main_transform


class BaseImageTestTimeAugmentation(BaseTestTimeAugmentation):
    """
    Abstract image base callback for test-time-augmentation.

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
    """

    def __init__(
            self,
            n: int,
            augmentations: List,
            augmentation_method: str = 'first',
            phase='val',
    ):
        super().__init__(n=n, augmentations=augmentations, augmentation_method=augmentation_method, phase=phase)
        self.transform = None
        self.inv_transform = None

    def __init_transform(self, trainer):
        self.inv_transform = []
        self.transform = []
        for dataloader_idx in range(len(trainer.__getattribute__(f'{self.phase}_dataloaders'))):
            self.inv_transform.append(
                inv_transform(
                    trainer.__getattribute__(
                        f'{self.phase}_dataloaders'
                    )[dataloader_idx].dataset.transform.transforms
                )
            )
            self.transform.append(
                main_transform(
                    trainer.__getattribute__(
                        f'{self.phase}_dataloaders'
                    )[dataloader_idx].dataset.transform.transforms
                )
            )

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
        self.__init_transform(trainer)

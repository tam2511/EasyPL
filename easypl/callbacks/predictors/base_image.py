from typing import List, Dict

from easypl.callbacks.predictors.base import BaseTestTimeAugmentation
from easypl.utilities.transforms import inv_transform, main_transform


class BaseImageTestTimeAugmentation(BaseTestTimeAugmentation):
    """ Image base callback for test-time-augmentation """

    def __init__(
            self,
            n: int,
            augmentations: List,
            augmentation_method: str = 'first',
            phase='val',
    ):
        """
        :param augmentations: list of augmentation transforms
        """
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
                    )[dataloader_idx].dataset.transform.transforms.transforms
                )
            )
            self.transform.append(
                main_transform(
                    trainer.__getattribute__(
                        f'{self.phase}_dataloaders'
                    )[dataloader_idx].dataset.transform.transforms.transforms
                )
            )

    def post_init(self, trainer, pl_module):
        self.__init_transform(trainer)

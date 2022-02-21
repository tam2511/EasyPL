import numpy as np
import torch
from easypl.callbacks.mixers.base import MixBaseCallback

AVAILABLE_DOMENS = ['classification', 'detection', 'segmentation']


class Mixup(MixBaseCallback):
    """Callback for mixuping"""

    def __init__(
            self,
            on_batch: bool = True,
            alpha: float = 0.4,
            num_workers: int = 1,
            p: float = 0.5,
            domen: str = 'classification'
    ):
        """
        :param on_batch: if True generate samples from batch else from dataset
        :param alpha: param for mixup operation
        :param num_workers: param for mixup operation
        :param p: mix probability
        :param domen: task name
        """
        super().__init__(on_batch=on_batch, samples_per=1, num_workers=num_workers, p=p)
        self.alpha = alpha
        if domen not in AVAILABLE_DOMENS:
            raise NotImplementedError(
                f'Domen {domen} is not supported in Mixup callback. Available domens: {AVAILABLE_DOMENS}')
        self.domen = domen

    def __mix_classificate(self, sample1: dict, sample2: dict, alpha: float) -> dict:
        return {
            key: sample1[key] * alpha + sample2[key] * (1 - alpha) if isinstance(sample1[key], torch.Tensor) else
            sample1[key] for key in sample1
        }

    def __mix_detection(self, sample1: dict, sample2: dict, alpha: float) -> dict:
        raise NotImplementedError

    def mix(self, sample1: dict, sample2: dict) -> dict:
        alpha = np.random.beta(self.alpha, self.alpha)
        if self.domen == 'classification':
            return self.__mix_classificate(sample1, sample2, alpha)
        elif self.domen == 'detection':
            return self.__mix_detection(sample1, sample2, alpha)
        elif self.domen == 'segmentation':
            return self.__mix_classificate(sample1, sample2, alpha)
        else:
            return sample1

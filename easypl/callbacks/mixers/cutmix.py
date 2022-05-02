from typing import Dict

import numpy as np
import torch
from easypl.callbacks.mixers.base import MixBaseCallback

AVAILABLE_DOMENS = ['classification', 'detection', 'segmentation']


class Cutmix(MixBaseCallback):
    """
    Callback for cutmixing data operations

    Attributes
    ----------
    on_batch: bool
        If True generate samples from batch otherwise from dataset.

    p: float
        Mix probability.

    num_workers: int
        Number of workers for mixing operation.

    alpha: float
        Parameter of cutmix operation.

    domen: str
        Name of task, in which will be mixed samples. Available: ["classification, segmentation"].
    """

    def __init__(
            self,
            on_batch: bool = True,
            alpha: float = 0.4,
            num_workers: int = 1,
            p: float = 0.5,
            domen: str = 'classification'
    ):
        super().__init__(on_batch=on_batch, samples_per=1, num_workers=num_workers, p=p)
        self.alpha = alpha
        if domen not in AVAILABLE_DOMENS:
            raise NotImplementedError(
                f'Domain {domen} is not supported in Mixup callback. Available domens: {AVAILABLE_DOMENS}')
        self.domen = domen

    def __random_bbox(self, height, width, alpha):
        ratio = torch.sqrt(1. - torch.tensor(alpha))
        w = width * ratio
        h = height * ratio

        # uniform
        cx = torch.rand(1)[0] * w
        cy = torch.rand(1)[0] * h
        x1 = torch.clip(cx - w // 2, torch.tensor(0), w).long()
        y1 = torch.clip(cy - h // 2, torch.tensor(0), h).long()
        x2 = torch.clip(cx + w // 2, torch.tensor(0), w).long()
        y2 = torch.clip(cy + h // 2, torch.tensor(0), h).long()
        return x1, y1, x2, y2

    def __mix_classificate(self, sample1: dict, sample2: dict, alpha: float) -> dict:
        data_key = self.data_keys[0]
        mix_sample = {key: sample1[key] for key in sample1}
        x1, y1, x2, y2 = self.__random_bbox(sample2[data_key].size(1), sample2[data_key].size(2), alpha)
        mix_sample[data_key][:, y1:y2, x1:x2] = sample2[data_key][:, y1:y2, x1:x2]
        alpha_ = 1 - (x2 - x1) * (y2 - y1) / (sample2[data_key].size(1) * sample2[data_key].size(2))
        for target_key in self.target_keys:
            mix_sample[target_key] = sample1[target_key] * alpha_ + sample2[target_key] * (1 - alpha_)
        return mix_sample

    def __mix_detection(self, sample1: dict, sample2: dict, alpha: float) -> dict:
        raise NotImplementedError

    def __mix_segmentation(self, sample1: dict, sample2: dict, alpha: float) -> dict:
        data_key = self.data_keys[0]
        mix_sample = {key: sample1[key] for key in sample1}
        x1, y1, x2, y2 = self.__random_bbox(sample2[data_key].size(1), sample2[data_key].size(2), alpha)
        mix_sample[data_key][:, y1:y2, x1:x2] = sample2[data_key][:, y1:y2, x1:x2]
        for target_key in self.target_keys:
            mix_sample[target_key][:, y1:y2, x1:x2] = sample2[target_key][:, y1:y2, x1:x2]
        return mix_sample

    def mix(
            self,
            sample1: Dict,
            sample2: Dict
    ) -> Dict:
        """
        Cutmix method for two samples.

        Attributes
        ----------
        sample1: Dict
            Sample of batch, which will be sampled with sample from `sample2`.

        sample2: Dict
            Sample from batch or dataset.

        Returns
        -------
        Dict
            Mixed sample.
        """
        if len(self.data_keys) != 1:
            raise NotImplementedError('Data keys must have len equal 1')
        sample2 = {key: sample2[key][0] for key in sample2}
        alpha = np.random.beta(self.alpha, self.alpha)
        if self.domen == 'classification':
            return self.__mix_classificate(sample1, sample2, alpha)
        elif self.domen == 'detection':
            return self.__mix_detection(sample1, sample2, alpha)
        elif self.domen == 'segmentation':
            return self.__mix_segmentation(sample1, sample2, alpha)
        else:
            return sample1

import numpy as np
from typing import Union, List
import torch
from easypl.callbacks.mixers.base import MixBaseCallback
from easypl.utilities.data import grids
from random import shuffle

AVAILABLE_DOMENS = ['classification', 'detection', 'segmentation']


class Mosaic(MixBaseCallback):
    """Callback for mosaic mixing"""

    def __init__(
            self,
            on_batch: bool = True,
            n_mosaics: Union[int, List[int]] = 4,
            num_workers: int = 1,
            p: float = 0.5,
            domen: str = 'classification'
    ):
        """
        :param on_batch: if True generate samples from batch else from dataset
        :param n_mosaics: number of images in mosaic grid
        :param num_workers: param for mixup operation
        :param p: mix probability
        :param domen: task name
        """
        n_mosaics = [_ - 1 for _ in n_mosaics] if isinstance(n_mosaics, list) else n_mosaics - 1
        super().__init__(on_batch=on_batch, samples_per=n_mosaics, num_workers=num_workers, p=p)
        if domen not in AVAILABLE_DOMENS:
            raise NotImplementedError(
                f'Domain {domen} is not supported in Mixup callback. Available domens: {AVAILABLE_DOMENS}')
        self.domen = domen

    def __random_bbox(self, height, width, height_bbox, width_bbox):
        cx = torch.tensor(np.random.randint(width_bbox // 2, width - width_bbox // 2))
        cy = torch.tensor(np.random.randint(height_bbox // 2, height - height_bbox // 2))
        x1 = torch.clip(cx - width_bbox // 2, 0, width_bbox).long()
        y1 = torch.clip(cy - height_bbox // 2, 0, height_bbox).long()
        x2 = torch.clip(cx + width_bbox // 2, 0, width_bbox).long()
        y2 = torch.clip(cy + height_bbox // 2, 0, height_bbox).long()
        return x1, y1, x2, y2

    def __mix_classificate(self, sample1: dict, samples: dict) -> dict:
        data_key = self.data_keys[0]
        if sample1[data_key].ndim != 3:
            raise ValueError(f'Tensor with key "{data_key}" must have 3 dims, but have {sample1[data_key].ndim}')
        h, w = sample1[data_key].shape[1:]
        n_samples = len(samples[data_key])

        grids_coords = grids(width=w, height=h, n=n_samples + 1)
        shuffle(grids_coords)
        mix_sample = {key: sample1[key] for key in sample1}
        for target_key in self.target_keys:
            mix_sample[target_key] *= (len(grids_coords) - n_samples) / len(grids_coords)
        for sample_idx in range(n_samples):
            (g_x1, g_x2), (g_y1, g_y2) = grids_coords[sample_idx]
            x1, y1, x2, y2 = self.__random_bbox(h, w, g_y2 - g_y1, g_x2 - g_x1)
            mix_sample[data_key][:, g_y1:g_y2, g_x1:g_x2] = samples[data_key][sample_idx][:, y1:y2, x1:x2]
            for target_key in self.target_keys:
                mix_sample[target_key] += samples[target_key][sample_idx] / len(grids_coords)
        return mix_sample

    def __mix_detection(self, sample1: dict, sample2: dict) -> dict:
        raise NotImplementedError

    def __mix_segmentation(self, sample1: dict, samples: dict) -> dict:
        data_key = self.data_keys[0]
        if sample1[data_key].ndim != 3:
            raise ValueError(f'Tensor with key "{data_key}" must have 3 dims, but have {sample1[data_key].ndim}')
        h, w = sample1[data_key].shape[1:]
        n_samples = len(samples[data_key])

        grids_coords = grids(width=w, height=h, n=n_samples + 1)
        shuffle(grids_coords)
        mix_sample = {key: sample1[key] for key in sample1}
        for sample_idx in range(n_samples):
            (g_x1, g_x2), (g_y1, g_y2) = grids_coords[sample_idx]
            x1, y1, x2, y2 = self.__random_bbox(h, w, g_y2 - g_y1, g_x2 - g_x1)
            mix_sample[data_key][:, g_y1:g_y2, g_x1:g_x2] = samples[data_key][sample_idx][:, y1:y2, x1:x2]
            for target_key in self.target_keys:
                mix_sample[target_key][:, g_y1:g_y2, g_x1:g_x2] = samples[target_key][sample_idx][:, y1:y2, x1:x2]
        return mix_sample

    def mix(self, sample1: dict, sample2: dict) -> dict:
        if len(self.data_keys) != 1:
            raise NotImplementedError('Data keys must have len equal 1')
        if self.domen == 'classification':
            return self.__mix_classificate(sample1, sample2)
        elif self.domen == 'detection':
            return self.__mix_detection(sample1, sample2)
        elif self.domen == 'segmentation':
            return self.__mix_segmentation(sample1, sample2)
        else:
            return sample1

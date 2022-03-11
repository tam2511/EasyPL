from typing import Union, List

from pytorch_lightning.callbacks import Callback
import numpy as np
import torch
from collections import Counter

from easypl.utilities.data import to_


class MixBaseCallback(Callback):
    """Abstract callback for mixing data operations"""

    def __init__(
            self,
            on_batch=True,
            samples_per: Union[int, List[int]] = 1,
            p: float = 0.5,
            num_workers: int = 1
    ):
        '''
        :param on_batch: if True generate samples from batch otherwise from dataset
        :param samples_per: number generating samples for one sample
        :param p: mix probability
        :param num_workers: number of workers for mixing operation
        '''
        super().__init__()
        self.on_batch = on_batch
        self.samples_per = samples_per
        self.p = p
        self.num_workers = num_workers

        self.data_keys = None
        self.target_keys = None
        self.device = torch.device('cpu')

    def __generate_dataset_sample(self, dataloader):
        dataset = dataloader.dataset.datasets
        sample = [dataset[i] for i in np.random.randint(0, len(dataset) - 1, self.samples_per)]
        sample = dataloader.loaders.collate_fn(sample)
        return to_(sample)

    def __generate_batch_sample(self, batch: dict, index_ignore: int = None):
        batch_size = self.__get_batch_size(batch)
        idxs = torch.from_numpy(
            np.random.choice([idx for idx in range(batch_size) if idx != index_ignore], self.samples_per))
        sample = {
            key: batch[key][idxs] if isinstance(batch[key], torch.Tensor) else [batch[key][idx] for idx in idxs]
            for key in batch
        }
        return to_(sample)

    def __check_device(self, batch):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                self.device = batch[key].device
                return

    def __get_batch_size(self, batch):
        batch_size = Counter([len(batch[key]) for key in batch]).most_common(1)[0][0]
        return batch_size

    def mix(self, sample1: dict, sample2: dict) -> dict:
        raise NotImplementedError

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.data_keys = pl_module.data_keys
        self.target_keys = pl_module.target_keys
        self.__check_device(batch)
        mix_idxs = np.where(np.random.uniform(size=self.__get_batch_size(batch)) < self.p)[0]
        # TODO: multi threading mixing in MixBaseCallback
        for key in self.target_keys:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].float()
        for idx in mix_idxs:
            sample1 = {key: batch[key][idx] for key in batch}
            sample2 = self.__generate_batch_sample(batch=batch, index_ignore=idx) \
                if self.on_batch else self.__generate_dataset_sample(trainer.train_dataloader)
            mix_sample = self.mix(sample1, sample2)
            for key in batch:
                batch[key][idx] = mix_sample[key]

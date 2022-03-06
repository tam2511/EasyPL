import os
import warnings
from typing import List, Dict

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.loggers import *

from easypl.callbacks.loggers.collector import ImageCollector


class BaseSampleLogger(Callback):
    def __init__(
            self,
            phase='train',
            max_samples=1,
            class_names=None,
            mode='first',
            sample_key=None,
            score_func=None,
            largest=True,
            dir_path=None,
            save_on_disk=False
    ):
        super().__init__()
        self.phase = phase
        self.max_samples = max_samples
        self.class_names = class_names
        self.mode = mode
        self.sample_key = sample_key
        self.score_func = score_func
        self.largest = largest
        self.dir_path = dir_path
        self.save_on_disk = save_on_disk

        self.is_init = False
        self.tag = f'{self.phase}/{self.mode}_predictions'
        self.collector = None
        self.data_keys = None
        self.logger = None
        self.epoch = 0

    def __sample(self, batch: dict, idx: int, dataloader_idx: int):
        if len(self.data_keys) > 1 and self.sample_key is None:
            raise ValueError('If "data_keys" includes more than 1 key, you should define "sample_key".')
        sample_key = self.data_keys[0] if len(self.data_keys) == 1 else self.sample_key
        sample = batch[sample_key][idx].cpu().numpy()
        return sample

    def __init_collectors(self, trainer):
        def get_collector(dataloader):
            if isinstance(dataloader, CombinedLoader):
                dataloader = dataloader.loaders
            bias = len(dataloader) % len(dataloader) if dataloader.drop_last and len(dataloader) > 0 else 0
            return ImageCollector(
                mode=self.mode,
                max_images=self.max_samples,
                score_func=self.score_func,
                largest=self.largest,
                dataset_size=len(dataloader.dataset) - bias)

        if self.phase == 'train':
            self.collector = get_collector(trainer.train_dataloader)
            return
        self.collector = []
        if self.phase == 'val':
            self.collector = []
            for dataloader_idx in range(len(trainer.val_dataloaders)):
                self.collector.append(get_collector(trainer.val_dataloaders[dataloader_idx]))
        elif self.phase == 'test':
            self.collector = []
            for dataloader_idx in range(len(trainer.test_dataloaders)):
                self.collector.append(get_collector(trainer.test_dataloaders[dataloader_idx]))
        elif self.phase == 'predict':
            self.collector = []
            for dataloader_idx in range(len(trainer.predict_dataloaders)):
                self.collector.append(get_collector(trainer.predict_dataloaders[dataloader_idx]))

    def __init_dir_path(self):
        if self.dir_path is None:
            root = os.path.join(os.getcwd(), 'lightning_logs')
            version_logs = [dir_name for dir_name in os.listdir(root) if os.path.isdir(os.path.join(root, dir_name))]
            last_version = max(map(lambda x: int(x.split('_')[-1]), version_logs))
            self.dir_path = os.path.join(root, 'version_{}'.format(last_version), 'images')
        os.makedirs(self.dir_path, exist_ok=True)

    def get_log(self, sample, output, target) -> Dict:
        raise NotImplementedError

    def _log_wandb(self, samples: list, dataloader_idx: int):
        raise NotImplementedError

    def _log_tensorboard(self, samples: list, dataloader_idx: int):
        raise NotImplementedError

    def _log_on_disk(self, samples: list, dataloader_idx: int):
        raise NotImplementedError

    def __log(self, samples: List, dataloader_idx: int = 0):
        if len(samples) == 0:
            return
        if isinstance(self.logger, WandbLogger):
            self._log_wandb(samples, dataloader_idx)
        elif isinstance(self.logger, TensorBoardLogger):
            self._log_tensorboard(samples, dataloader_idx)
        else:
            warnings.warn(f'{self.logger.__class__.__name__} is not supported. Samples will log on disk.', Warning,
                          stacklevel=2)
            self.save_on_disk = True

        if self.save_on_disk:
            self._log_on_disk(samples, dataloader_idx)

    def on_train_start(self, trainer, pl_module):
        pl_module.return_output_phase[self.phase] = True

    def __main_post_init(self, trainer, pl_module):
        if self.logger is None:
            self.logger = trainer.logger
        if self.collector is None:
            self.__init_collectors(trainer)
        self.__init_dir_path()
        self.data_keys = pl_module.data_keys

    def _post_init(self, trainer, pl_module):
        raise NotImplementedError

    def __on_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
    ):
        if not self.is_init:
            self.__main_post_init(trainer, pl_module)
            self._post_init(trainer, pl_module)
            self.is_init = True

        output = outputs['output']
        target = outputs['target']
        for i in range(len(output)):
            sample = self.__sample(batch, i, dataloader_idx)
            if isinstance(self.collector, list):
                self.collector[dataloader_idx].update(output[i], target[i], sample)
            else:
                self.collector.update(output[i], target[i], sample)

    def on_train_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
    ):
        if self.phase == 'train':
            self.__on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
    ):
        if self.phase == 'val':
            self.__on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_predict_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
    ):
        if self.phase == 'predict':
            self.__on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
    ):
        if self.phase == 'test':
            self.__on_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def __on_epoch_end(
            self,
            trainer,
            pl_module,
            unused=None
    ):
        if isinstance(self.collector, list):
            for dataloader_idx in range(len(self.collector)):
                results = self.collector[dataloader_idx].compute()
                samples = [self.get_log(result['data'], result['output'], result['target']) for result in results]
                self.__log(samples, dataloader_idx)
                self.collector[dataloader_idx].reset()
        else:
            results = self.collector.compute()
            samples = [self.get_log(result['data'], result['output'], result['target']) for result in results]
            self.__log(samples)
            self.collector.reset()

    def on_train_epoch_end(
            self,
            trainer,
            pl_module,
            unused=None
    ):
        if self.phase == 'train':
            self.__on_epoch_end(trainer, pl_module, unused=unused)

    def on_validation_epoch_end(
            self,
            trainer,
            pl_module,
            unused=None
    ):
        if self.phase == 'val':
            self.__on_epoch_end(trainer, pl_module, unused=unused)

    def on_test_epoch_end(
            self,
            trainer,
            pl_module,
            unused=None
    ):
        if self.phase == 'test':
            self.__on_epoch_end(trainer, pl_module, unused=unused)

    def on_predict_epoch_end(
            self,
            trainer,
            pl_module,
            unused=None
    ):
        if self.phase == 'predict':
            self.__on_epoch_end(trainer, pl_module, unused=unused)

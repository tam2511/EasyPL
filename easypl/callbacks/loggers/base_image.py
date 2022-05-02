from typing import Optional, List, Callable

import pytorch_lightning

from easypl.utilities.transforms import inv_transform
from easypl.callbacks.loggers.base import BaseSampleLogger


class BaseImageLogger(BaseSampleLogger):
    """

    Abstract callback class for logging images

    Attributes
    ----------
    phase: str
        Phase which will be used by this Logger.
        Available: ["train", "val", "test", "predict"].

    max_samples: int
        Maximum number of samples which will be logged at one epoch.

    class_names: Optional[List]
        List of class names for pretty logging.
        If None, then class_names will set range of number of classes.

    mode: str
        Mode of sample generation.
        Available modes: ["random", "first", "top"].

    sample_key: Optional
        Key of batch, which define sample.
        If None, then sample_key will parse `learner.data_keys`.

    score_func: Optional[Callable]
        Function for score evaluation. Necessary if "mode" = "top".

    largest: bool
        Sorting order for "top" mode

    dir_path: Optional[str]
        If defined, then logs will be writed in this directory. Else in lighting_logs.

    save_on_disk: bool
        If true, then logs will be writed on disk to "dir_path".

    """
    def __init__(
            self,
            phase: str = 'train',
            max_samples: int = 1,
            class_names: Optional[List] = None,
            mode: str = 'first',
            sample_key: Optional = None,
            score_func: Optional[Callable] = None,
            largest: bool = True,
            dir_path: Optional[str] = None,
            save_on_disk: bool = False
    ):
        super().__init__(
            phase=phase,
            max_samples=max_samples,
            class_names=class_names,
            mode=mode,
            sample_key=sample_key,
            score_func=score_func,
            largest=largest,
            dir_path=dir_path,
            save_on_disk=save_on_disk
        )
        self.inv_transform = None

    def __init_transform(self, trainer):
        if self.phase == 'train':
            self.inv_transform = [inv_transform(
                trainer.train_dataloader.dataset.datasets.transform.transforms.transforms)]
            return
        self.inv_transform = []
        for dataloader_idx in range(len(trainer.__getattribute__(f'{self.phase}_dataloaders'))):
            self.inv_transform.append(
                inv_transform(
                    trainer.__getattribute__(
                        f'{self.phase}_dataloaders'
                    )[dataloader_idx].dataset.transform.transforms.transforms
                )
            )

    def _post_init(
            self,
            trainer: pytorch_lightning.Trainer,
            pl_module: pytorch_lightning.LightningModule
    ):
        """
        Method for initialization inverse transforms.

        Attributes
        ----------
        trainer: pytorch_lightning.Trainer
            Trainer of pytorch-lightning

        pl_module: pytorch_lightning.LightningModule
            LightningModule of pytorch-lightning

        """
        self.__init_transform(trainer)

from easypl.utilities.transforms import inv_transform
from easypl.callbacks.loggers.base import BaseSampleLogger


class BaseImageLogger(BaseSampleLogger):
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
        if self.phase == 'val':
            for dataloader_idx in range(len(trainer.val_dataloaders)):
                self.inv_transform.append(
                    inv_transform(
                        trainer.val_dataloaders[dataloader_idx].dataset.transform.transforms.transforms
                    )
                )
        if self.phase == 'test':
            for dataloader_idx in range(len(trainer.test_dataloaders)):
                self.inv_transform.append(
                    inv_transform(
                        trainer.test_dataloaders[dataloader_idx].dataset.transform.transforms.transforms
                    )
                )
        if self.phase == 'predict':
            for dataloader_idx in range(len(trainer.predict_dataloaders)):
                self.inv_transform.append(
                    inv_transform(
                        trainer.predict_dataloaders[dataloader_idx].dataset.transform.transforms.transforms
                    )
                )

    def _post_init(self, trainer, pl_module):
        self.__init_transform(trainer)

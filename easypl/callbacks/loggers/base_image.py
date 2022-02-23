from pytorch_lightning.callbacks import Callback

from easypl.utilities.transforms import inv_transform
from easypl.callbacks.loggers.collector import ImageCollector


class ImageLogger(Callback):
    def __init__(
            self,
            phase='train',
            max_images=1,
            class_names=None,
            mode='first',
            image_key=None,
            score_func=None,
            largest=True
    ):
        super().__init__()
        self.phase = phase
        self.max_images = max_images
        self.class_names = class_names
        self.mode = mode
        self.image_key = image_key
        self.score_func = score_func
        self.largest = largest

        self.collector = None
        self.inv_transform = None
        self.data_keys = None

    def __image(self, batch: dict, idx: int, dataloader_idx: int):
        if len(self.data_keys) > 1 and self.image_key is None:
            raise ValueError('If "data_keys" includes more than 1 key, you should define "image_key".')
        image_key = self.data_keys[0] if len(self.data_keys) == 1 else self.image_key
        image = batch[image_key][idx]
        if self.phase == 'train':
            return self.inv_transform(image)
        else:
            return self.inv_transform[dataloader_idx](image)

    def __init_transform(self, trainer):
        if self.mode == 'train':
            self.inv_transform = inv_transform(
                trainer.train_dataloader.dataset.datasets.transform.transforms.transforms)
            return
        self.inv_transform = []
        if self.mode == 'val':
            for dataloader_idx in range(len(trainer.val_dataloaders)):
                self.inv_transform.append(
                    inv_transform(
                        trainer.val_dataloaders[dataloader_idx].dataset.datasets.transform.transforms.transforms
                    )
                )
        if self.mode == 'test':
            for dataloader_idx in range(len(trainer.test_dataloaders)):
                self.inv_transform.append(
                    inv_transform(
                        trainer.test_dataloaders[dataloader_idx].dataset.datasets.transform.transforms.transforms
                    )
                )
        if self.mode == 'predict':
            for dataloader_idx in range(len(trainer.predict_dataloaders)):
                self.inv_transform.append(
                    inv_transform(
                        trainer.predict_dataloaders[dataloader_idx].dataset.datasets.transform.transforms.transforms
                    )
                )

    def __init_collectors(self, trainer):
        def get_collector(dataloader):
            bias = len(dataloader) % len(dataloader) if dataloader.drop_last and len(dataloader) > 0 else 0
            return ImageCollector(
                mode=self.mode,
                max_images=self.max_images,
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

    def draw(self, image, output, target):
        raise NotImplementedError

    def __log(self, image):
        ...

    def on_train_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
    ):
        if self.collector is None:
            self.__init_collectors(trainer)
        if self.inv_transform is None:
            self.__init_transform(trainer)
        self.data_keys = pl_module.data_keys
        output = outputs['output']
        target = outputs['target']
        for i in range(len(output)):
            image = self.__image(batch, i, dataloader_idx)
            results = self.collector.update(output[i], target[i], image)
            if len(results) > 0:
                for result in results:
                    drawed_image = self.draw(result['data'], result['output'], result['target'])
                    self.__log(drawed_image)

    def on_train_epoch_end(
            self,
            trainer,
            pl_module,
            unused=None
    ):
        results = self.collector.compute()
        if len(results) > 0:
            for result in results:
                drawed_image = self.draw(result['data'], result['output'], result['target'])
                self.__log(drawed_image)
        self.collector.reset()

from typing import Optional, Any
from albumentations.augmentations.transforms import Normalize
from albumentations.core.composition import Compose

from pytorch_lightning.callbacks import Callback

AVAILABLE_MODES = ['first', 'random', 'top']


class ImageLogger(Callback):
    def __init__(
            self,
            phase='train',
            max_images=1,
            class_names=None,
            mode='first',
            image_key=None
    ):
        super().__init__()
        self.phase = phase
        self.max_images = max_images
        self.class_names = class_names
        self.mode = mode
        self.image_key = image_key

        self.data_keys = None

    def __image(self, batch: dict, idx: int):
        if len(self.data_keys) > 1 and self.image_key is None:
            raise ValueError('If "data_keys" includes more than 1 key, you should define "image_key".')
        image_key = self.data_keys[0] if len(self.data_keys) == 1 else self.image_key
        return batch[image_key][idx]

    def on_train_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
    ):
        self.data_keys = pl_module.data_keys
        output = outputs['output']
        target = outputs['target']

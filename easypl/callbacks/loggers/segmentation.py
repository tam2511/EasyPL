import warnings
from typing import Dict
import numpy as np
import torch
import colorsys
import random
from uuid import uuid4
import cv2
import os

from easypl.callbacks.loggers.base_image import BaseImageLogger


class SegmentationImageLogger(BaseImageLogger):
    def __init__(
        self,
        phase='train',
        max_samples=1,
        class_names=None,
        num_classes=None,
        mode='first',
        sample_key=None,
        score_func=None,
        largest=True,
        dir_path=None,
        save_on_disk=False,
        background_class=0
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
        self.num_classes = num_classes
        if self.class_names is None:
            if self.num_classes is None:
                raise ValueError('If "class_names" is None then you should define num_classes')
            else:
                self.class_names = [f'class_{i}' for i in range(self.num_classes)]
        if self.num_classes is None:
            self.num_classes = len(self.class_names)
        self.colors = self.__generate_colors()
        self.background_class = background_class

    def get_log(self, sample, output, target) -> Dict:
        image = self.inv_transform(image=sample)['image'].astype('uint8')
        if not isinstance(output, torch.Tensor):
            raise ValueError('Output must be torch.Tensor type.')
        if not isinstance(target, torch.Tensor):
            raise ValueError('Target must be torch.Tensor type.')
        if output.ndim == 2:
            pred_mask = output.cpu().numpy().astype('int32')
        elif output.ndim == 3:
            pred_mask = output.argmax(dim=0).cpu().numpy().astype('int32')
        else:
            raise ValueError(f'Output must to have 2 or 3 dims (but have {output.ndim}!).')
        if target.ndim == 2:
            target_mask = target.cpu().numpy().astype('int32')
        else:
            raise ValueError(f'Target must to have 2 dims (but have {target.ndim}!).')
        return {
            'image': image,
            'pred_mask': pred_mask,
            'target_mask': target_mask
        }

    def _log_wandb(self, samples: list, dataloader_idx: int):
        images = [_['image'] for _ in samples]
        masks = [
            {
                "predictions": {
                    "mask_data": _['pred_mask'],
                    "class_labels": {i: self.class_names[i] for i in range(len(self.class_names))}
                },
                "ground_truth": {
                    "mask_data": _['target_mask'],
                    "class_labels": {i: self.class_names[i] for i in range(len(self.class_names))}
                },
            }
            for _ in samples
        ]
        self.logger.log_image(key=self.tag, images=images, masks=masks)

    def _log_tensorboard(self, samples: list, dataloader_idx: int):
        warnings.warn(f'TensorboardLogger does not supported. Images will save on disk', Warning, stacklevel=2)
        self.save_on_disk = True

    def __generate_colors(self):
        return np.random.randint(0, 255, (self.num_classes, 3))

    def _log_on_disk(self, samples: list, dataloader_idx: int):
        for i in range(len(samples)):
            pred_mask, target_mask = np.copy(samples[i]['image']), np.copy(samples[i]['image'])
            color_mask = np.zeros_like(samples[i]['image'])
            for class_idx in range(self.num_classes):
                if class_idx == self.background_class:
                    continue
                for color_i in range(3):
                    color_mask[:, :, color_i].fill(self.colors[class_idx][color_i])
                pred_mask = np.where(
                    (samples[i]['pred_mask'] == class_idx).repeat(3, -1).reshape(*samples[i]['pred_mask'].shape, 3),
                    color_mask,
                    pred_mask
                )
                target_mask = np.where(
                    (samples[i]['target_mask'] == class_idx).repeat(3, -1).reshape(*samples[i]['target_mask'].shape, 3),
                    color_mask,
                    target_mask
                )
            pred_mask = (pred_mask + samples[i]['image']) // 2
            target_mask = (target_mask + samples[i]['image']) // 2

            dest_dir = os.path.join(self.dir_path, f'epoch_{self.epoch}', self.phase)
            dest_dir = os.path.join(dest_dir, f'dataloader_{dataloader_idx}')
            os.makedirs(dest_dir, exist_ok=True)
            guid = str(uuid4())
            cv2.imwrite(os.path.join(dest_dir, f'{guid}_pred.jpg'), cv2.cvtColor(pred_mask, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(dest_dir, f'{guid}_gt.jpg'), cv2.cvtColor(target_mask, cv2.COLOR_RGB2BGR))

from typing import Dict

import torch

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
        self.num_classes = num_classes
        if self.class_names is None:
            if self.num_classes is None:
                raise ValueError('If "class_names" is None then you should define num_classes')
            else:
                self.class_names = [f'class_{i}' for i in range(self.num_classes)]

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

    def _log_wandb(self, samples: list):
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

    def _log_tensorboard(self, samples: list):
        raise NotImplementedError

    def _log_on_disk(self, samples: list):
        raise NotImplementedError

from typing import Dict

import torch

from easypl.callbacks.loggers.base_image import BaseImageLogger


class SegmentationImageLogger(BaseImageLogger):
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

    def get_log(self, sample, output, target) -> Dict:
        image = self.inv_transform(image=sample)['image']
        if not isinstance(output, torch.Tensor):
            raise ValueError('Output must be torch.Tensor type.')
        if not isinstance(target, torch.Tensor):
            raise ValueError('Target must be torch.Tensor type.')
        if output.ndim == 2:
            pred_mask = output.numpy().astype('int32')
        elif output.ndim == 3:
            pred_mask = output.argmax(dim=0).numpy().astype('int32')
        else:
            raise ValueError(f'Output must to have 2 or 3 dims (but have {output.ndim}!).')
        if target.ndim == 2:
            target_mask = target.numpy().astype('int32')
        else:
            raise ValueError(f'Target must to have 2 dims (but have {target.ndim}!).')
        return {
            'image': image,
            'pred_mask': pred_mask,
            'target_mask': target_mask
        }

    def _log_wandb(self, samples: list):
        raise NotImplementedError

    def _log_tensorboard(self, samples: list):
        raise NotImplementedError

    def _log_on_disk(self, samples: list):
        raise NotImplementedError

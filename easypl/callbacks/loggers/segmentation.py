import warnings
from typing import Dict
import numpy as np
import torch
import colorsys
import random
from uuid import uuid4
import cv2
import os
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from easypl.callbacks.loggers.base_image import BaseImageLogger


class SegmentationImageLogger(BaseImageLogger):
    def __init__(
            self,
            phase='train',
            max_samples=1,
            class_names=None,
            num_classes=None,
            max_log_classes=None,
            mode='first',
            sample_key=None,
            score_func=None,
            largest=True,
            dir_path=None,
            save_on_disk=False,
            background_class=0,
            dpi=100
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
        self.dpi = dpi
        self.max_log_classes = self.num_classes if max_log_classes is None else min(max_log_classes, self.num_classes)

        self.pad = 20

    def get_log(self, sample, output, target, dataloader_idx=0) -> Dict:
        image = self.inv_transform[dataloader_idx](image=sample)['image'].astype('uint8')
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
        elif target.ndim == 3:
            # TODO multilabel segmentation mask in image_logger
            target_mask = np.ones(
                target.shape[1:],
                dtype='uint8'
            ) * self.background_class if self.background_class >= 0 else np.zeros(target.shape[1:], dtype='uint8')
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
        self.logger.log_image(key=f'{self.tag}_dataloader {dataloader_idx}', images=images, masks=masks)

    def _log_tensorboard(self, samples: list, dataloader_idx: int):
        warnings.warn(f'TensorboardLogger does not supported. Images will save on disk', Warning, stacklevel=2)
        self.save_on_disk = True

    def __generate_colors(self):
        hsv = [(i / self.num_classes, 1, 1) for i in range(self.num_classes)]
        colors = list(map(lambda c: np.array(colorsys.hsv_to_rgb(*c)), hsv))
        random.shuffle(colors)
        return (np.stack(colors) * 255).astype('uint8')

    def __get_legend(self, classes_idxs):
        classes = [self.class_names[idx] for idx in classes_idxs if idx != self.background_class]
        colors = [self.colors[idx] for idx in classes_idxs if idx != self.background_class]
        handles = [Rectangle((0, 0), 1, 1, color=tuple([_ / 255 for _ in color])) for color in colors]
        #     stage 1
        fig = plt.figure(dpi=self.dpi)
        canvas = FigureCanvas(fig)
        legend = fig.legend(handles, classes, loc='upper center')
        canvas.draw()
        w, h = legend.get_window_extent().width, legend.get_window_extent().height
        plt.close(fig)
        #     stage 2
        fig = plt.figure(figsize=((w + self.pad) / 100, (h + self.pad) / 100), dpi=self.dpi)
        canvas = FigureCanvas(fig)
        fig.legend(handles, classes, loc='upper center')
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image

    def __add_legend(self, image, classes_idxs):
        legend = self.__get_legend(classes_idxs)
        image_h, image_w, _ = image.shape
        legend_h, legend_w, _ = legend.shape
        w = max(image_w, legend_w) + 2 * self.pad
        x_image = (w - image_w) // 2
        x_legend = (w - legend_w) // 2
        result = np.ones((image_h + legend_h + 3 * self.pad, w, 3), dtype=np.uint8) * 255
        result[self.pad:image_h + self.pad, x_image:x_image + image_w, :] = image
        result[image_h + 2 * self.pad: image_h + legend_h + 2 * self.pad, x_legend:x_legend + legend_w, :] = legend
        result = cv2.rectangle(result, (x_image, self.pad), (x_image + image_w, image_h + self.pad), (0, 0, 0), 2)
        return result

    def __merge_masks(self, sample, pred_mask, target_mask):
        pred_mask = cv2.addWeighted(pred_mask, 0.5, sample['image'], 0.5, 0.0)
        target_mask = cv2.addWeighted(target_mask, 0.5, sample['image'], 0.5, 0.0)
        pred_class_idxs, _ = np.unique(sample['pred_mask'], return_counts=True)
        pred_class_idxs = [class_idx for class_idx in pred_class_idxs if class_idx != self.background_class]
        target_class_idxs, _ = np.unique(sample['target_mask'], return_counts=True)
        target_class_idxs = [class_idx for class_idx in target_class_idxs if class_idx != self.background_class]
        pred_mask = self.__add_legend(pred_mask, pred_class_idxs[:self.max_log_classes])
        target_mask = self.__add_legend(target_mask, target_class_idxs[:self.max_log_classes])
        pred_h, pred_w, _ = pred_mask.shape
        target_h, target_w, _ = target_mask.shape
        result = np.ones((max(pred_h, target_h), pred_w + target_w, 3), dtype='uint8') * 255
        result[:pred_h, :pred_w, :] = pred_mask
        result[:target_h, pred_w:pred_w + target_w, :] = target_mask
        return result

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
            result_image = self.__merge_masks(samples[i], pred_mask, target_mask)
            dest_dir = os.path.join(self.dir_path, f'epoch_{self.epoch}', self.phase)
            dest_dir = os.path.join(dest_dir, f'dataloader_{dataloader_idx}')
            os.makedirs(dest_dir, exist_ok=True)
            guid = str(uuid4())
            cv2.imwrite(os.path.join(dest_dir, f'{guid}.jpg'), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

import warnings
from typing import Dict, Optional, Callable, List
import numpy as np
import torch
from torch.nn.functional import one_hot
import pandas as pd
from uuid import uuid4
import cv2
import os
from pandas.plotting import table
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from easypl.callbacks.loggers.base_image import BaseImageLogger


class ClassificationImageLogger(BaseImageLogger):
    """

    Callback class for logging images in classification task

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

    num_classes: Optional[int]
        Number of classes. Necessary if `class_names` is None.

    max_log_classes: Optional[int]
        Max of number classes, which will be logged in one sample.

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
            num_classes: Optional[int] = None,
            max_log_classes: Optional[int] = None,
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
        self.num_classes = num_classes
        if self.class_names is None:
            if self.num_classes is None:
                raise ValueError('If "class_names" is None then you should define num_classes')
            else:
                self.class_names = [f'class_{i}' for i in range(self.num_classes)]
        if self.num_classes is None:
            self.num_classes = len(self.class_names)
        self.max_log_classes = self.num_classes if max_log_classes is None else min(max_log_classes, self.num_classes)

        self.pad = 20

    def get_log(
        self,
        sample: np.ndarray,
        output: torch.Tensor,
        target: torch.Tensor,
        dataloader_idx: int = 0
    ) -> Dict:
        """
        Method for preparing data for image classification logging

        Attributes
        ----------
        sample: np.ndarray
            Image in numpy ndarray format.

        output: torch.Tensor
            Output of model in Tensor format.

        target: torch.Tensor
            Target of record of dataset in Tensor format.

        dataloader_idx: int, default: 0
            Index of dataloader.

        Returns
        -------
        Dict
            Dict with `image`, `preds` and `targets` keys.

        """
        image = self.inv_transform[dataloader_idx](image=sample)['image'].astype('uint8')
        if not isinstance(output, torch.Tensor):
            raise ValueError('Output must be torch.Tensor type.')
        if not isinstance(target, torch.Tensor):
            raise ValueError('Target must be torch.Tensor type.')
        if output.ndim == 1:
            preds = output.cpu()
        else:
            raise ValueError(f'Output must to have 1 dim (but have {output.ndim}!).')
        if target.ndim == 1:
            targets = target.cpu()
        elif target.ndim == 0:
            targets = one_hot(target, num_classes=self.num_classes).cpu()
        else:
            raise ValueError(f'Target must to have 0 or 1 dims (but have {target.ndim}!).')
        return {
            'image': image,
            'preds': preds,
            'targets': targets
        }

    def _log_wandb(
            self,
            samples: List,
            dataloader_idx: int = 0
    ):
        """

        Method for wandb logging.

        Attributes
        ----------
        samples: List
            List of returns from `get_log`.

        dataloader_idx: int, default: 0
            Index of dataloader.

        """
        images = [self.__add_table(sample['image'], sample['preds'], sample['targets']) for sample in samples]
        self.logger.log_image(key=f'{self.tag}_dataloader {dataloader_idx}', images=images)

    def _log_tensorboard(
            self,
            samples: List,
            dataloader_idx: int = 0
    ):
        """

        Method for tensorboard logging.

        Attributes
        ----------
        samples: List
            List of returns from `get_log`.

        dataloader_idx: int, default: 0
            Index of dataloader.

        """
        warnings.warn(f'TensorboardLogger does not supported. Images will save on disk', Warning, stacklevel=2)
        self.save_on_disk = True

    def __get_table(self, preds: torch.Tensor, targets: torch.Tensor):
        _, preds_class = torch.topk(preds.float(), dim=0, k=self.max_log_classes, largest=True)
        _, targets_class = torch.topk(targets.float(), dim=0, k=self.max_log_classes, largest=True)
        targets_class = torch.tensor([class_idx for class_idx in targets_class if class_idx not in preds_class])
        class_idxs = torch.cat([preds_class, targets_class]).long()
        classes = [self.class_names[idx] for idx in class_idxs]
        data = [{'pred': preds[idx].item(), 'ground truth': targets[idx].item()} for idx in class_idxs]
        dt = pd.DataFrame(data, index=classes)
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca(frame_on=False)
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        table(ax, dt, cellLoc='center', rowLoc='center', loc='center')  # where df is your data frame
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image

    def __add_table(self, image, preds, targets):
        table_image = self.__get_table(preds, targets)
        image_h, image_w, _ = image.shape
        legend_h, legend_w, _ = table_image.shape
        w = max(image_w, legend_w) + 2 * self.pad
        x_image = (w - image_w) // 2
        x_legend = (w - legend_w) // 2
        result = np.ones((image_h + legend_h + 3 * self.pad, w, 3), dtype=np.uint8) * 255
        result[self.pad:image_h + self.pad, x_image:x_image + image_w, :] = image
        result[image_h + 2 * self.pad: image_h + legend_h + 2 * self.pad, x_legend:x_legend + legend_w, :] = table_image
        result = cv2.rectangle(result, (x_image, self.pad), (x_image + image_w, image_h + self.pad), (0, 0, 0), 2)
        return result

    def _log_on_disk(
            self,
            samples: List,
            dataloader_idx: int = 0
    ):
        """

        Method for logging on disk.

        Attributes
        ----------
        samples: List
            List of returns from `get_log`.

        dataloader_idx: int, default: 0
            Index of dataloader.

        """
        for i in range(len(samples)):
            result_image = self.__add_table(samples[i]['image'], samples[i]['preds'], samples[i]['targets'])
            dest_dir = os.path.join(self.dir_path, f'epoch_{self.epoch}', self.phase)
            dest_dir = os.path.join(dest_dir, f'dataloader_{dataloader_idx}')
            os.makedirs(dest_dir, exist_ok=True)
            guid = str(uuid4())
            cv2.imwrite(os.path.join(dest_dir, f'{guid}.jpg'), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

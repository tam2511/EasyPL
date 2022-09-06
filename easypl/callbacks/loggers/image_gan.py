import warnings
from typing import Dict, Optional, Callable, List, Tuple, Any
import numpy as np
import torch
from matplotlib.patches import Rectangle
from uuid import uuid4
import cv2
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from easypl.callbacks.loggers.base_image import BaseImageLogger

THICKNESS = 2
PAD = 10
PRED_COLOR = (0, 255, 0)
TARGET_COLOR = (255, 0, 0)


class GANImageLogger(BaseImageLogger):
    """

    Callback class for logging images in gan task

    Attributes
    ----------
    phase: str
        Phase which will be used by this Logger.
        Available: ["train", "val", "test", "predict"].

    max_samples: int
        Maximum number of samples which will be logged at one epoch.

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
            mode: str = 'first',
            sample_key: Optional = None,
            score_func: Optional[Callable] = None,
            largest: bool = True,
            dir_path: Optional[str] = None,
            save_on_disk: bool = False,
            dpi=100
    ):
        super().__init__(
            phase=phase,
            max_samples=max_samples,
            mode=mode,
            sample_key=sample_key,
            score_func=score_func,
            largest=largest,
            dir_path=dir_path,
            save_on_disk=save_on_disk
        )
        self.dpi = dpi
        self.pad = 20

    def get_log(
            self,
            sample: Any,
            output: torch.Tensor,
            target: torch.Tensor,
            dataloader_idx: int = 0
    ) -> Dict:
        """
        Method for preparing data for image logging

        Attributes
        ----------
        sample: np.ndarray
            Predicted image in numpy ndarray format.

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
        if target.shape != output.shape:
            raise ValueError(
                f'Output and target images must have same shape (but output shape is {output.shape} and target shape is {target.shape}!).'
            )
        gt_image = self.inv_transform[dataloader_idx](image=target.cpu().numpy())['image'].astype('uint8')
        pred_image = self.inv_transform[dataloader_idx](image=output.cpu().numpy())['image'].astype('uint8')

        return {
            'image': gt_image,
            'preds': pred_image,
            'targets': None
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
        images = [self.draw_pair(sample['image'], sample['preds']) for sample in samples]
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

    def draw_pair(
            self,
            gt_image: np.ndarray,
            pred_image: np.ndarray
    ) -> np.ndarray:
        h, w = gt_image.shape[:2]
        fig = plt.figure(figsize=((2 * w + 4 * self.pad) / 100, (h + self.pad) / 100), dpi=self.dpi)
        canvas = FigureCanvas(fig)
        fig.add_subplot(1, 2, 1)
        plt.imshow(gt_image)
        plt.axis('off')
        plt.title("Ground truth")

        fig.add_subplot(1, 2, 2)
        plt.imshow(pred_image)
        plt.axis('off')
        plt.title("Prediction")
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image

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
            result_image = self.draw_pair(samples[i]['image'], samples[i]['preds'])
            dest_dir = os.path.join(self.dir_path, f'epoch_{self.epoch}', self.phase)
            dest_dir = os.path.join(dest_dir, f'dataloader_{dataloader_idx}')
            os.makedirs(dest_dir, exist_ok=True)
            guid = str(uuid4())
            cv2.imwrite(os.path.join(dest_dir, f'{guid}.jpg'), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

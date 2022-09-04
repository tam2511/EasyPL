import warnings
from typing import Dict, Optional, Callable, List, Tuple
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


class DetectionImageLogger(BaseImageLogger):
    """

    Callback class for logging images in detection task

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

    max_detections_per_image: Optional[int]
        Max of number detections, which will be logged in one sample.

    confidence: Optional[float]
        Min confidence of predictions, which will be logged in one sample.

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
            max_detections_per_image: Optional[int] = None,
            confidence: Optional[float] = None,
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
        self.confidence = 0 if confidence is None else confidence
        self.max_detections_per_image = -1 if max_detections_per_image is None else max_detections_per_image
        self.dpi = dpi
        self.pad = 20

    def get_log(
            self,
            sample: np.ndarray,
            output: torch.Tensor,
            target: torch.Tensor,
            dataloader_idx: int = 0
    ) -> Dict:
        """
        Method for preparing data for image detection logging

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

        if output.ndim == 2:
            preds = output.cpu()
        else:
            raise ValueError(f'Output must to have 1 dim (but have {output.ndim}!).')
        if target.ndim == 2:
            targets = target.cpu()
        else:
            raise ValueError(f'Target must to have 0 or 1 dims (but have {target.ndim}!).')
        preds = preds[torch.where(preds[:, 4] > self.confidence)[0]]
        preds = preds[preds[:, 4].sort()[1][:self.max_detections_per_image]]
        targets = targets[torch.where(targets[:, 4] >= 0)[0]]
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
        images = [self.__draw_detections(sample['image'], sample['preds'], sample['targets']) for sample in samples]
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

    def __get_legend(self):
        classes = ['prediction', 'ground thruth']
        colors = [PRED_COLOR, TARGET_COLOR]
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

    def __add_legend(self, image):
        legend = self.__get_legend()
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

    def __draw_detection(
            self,
            image: np.ndarray,
            box: Tuple,
            message: Optional[str] = None,
            color: Tuple = (255, 0, 0),
    ):
        image = cv2.rectange(image, box[:2], box[2:], color=color, thickness=THICKNESS)
        if message is None:
            return image
        image = cv2.putText(
            image,
            message,
            (box[0], box[1] - PAD) if box[1] > PAD else (box[0], box[3] + PAD),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            THICKNESS,
            cv2.LINE_AA
        )
        return image

    def __draw_detections(self, image: np.ndarray, preds: torch.Tensor, targets: torch.Tensor):
        image_ = image.copy()
        for pred in preds:
            x1, y1, x2, y2 = int(pred[0].item()), int(pred[1].item()), int(pred[2].item()), int(pred[3].item())
            prob = float(pred[4].item())
            class_idx = int(pred[5].item())
            class_name = self.class_names[class_idx]
            message = '{}: {:.4f}'.format(class_name, prob)
            image_ = self.__draw_detection(
                image_,
                (x1, y1, x2, y2),
                message,
                PRED_COLOR
            )

        for target in targets:
            x1, y1, x2, y2 = int(target[0].item()), int(target[1].item()), int(target[2].item()), int(target[3].item())
            class_idx = int(target[4].item())
            message = self.class_names[class_idx]
            image_ = self.__draw_detection(
                image_,
                (x1, y1, x2, y2),
                message,
                TARGET_COLOR
            )
        return self.__add_legend(image_)

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
            result_image = self.__draw_detections(samples[i]['image'], samples[i]['preds'], samples[i]['targets'])
            dest_dir = os.path.join(self.dir_path, f'epoch_{self.epoch}', self.phase)
            dest_dir = os.path.join(dest_dir, f'dataloader_{dataloader_idx}')
            os.makedirs(dest_dir, exist_ok=True)
            guid = str(uuid4())
            cv2.imwrite(os.path.join(dest_dir, f'{guid}.jpg'), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

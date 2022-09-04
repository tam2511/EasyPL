import json
from typing import Callable, Optional, Dict
import pandas as pd
import numpy as np

from easypl.datasets.base import PathBaseDataset


class CSVDatasetDetection(PathBaseDataset):
    """
        Csv dataset for detection

        Attributes
        ----------
        csv_path: str
            path to csv file with paths of images

        return_label: bool
            if True return dict with two keys (image, annotations), else return dict with one key (image)

        image_column: Optional[str]
            column name or None. If None then will be getting the first column

        target_column: Optional[str]
            column name/names or None. If None then will be getting all but the second column

        image_prefix: str
            path prefix which will be added to paths of images in csv file

        path_transform: Optional[Callable]
            None or function for transform of path. Will be os.path.join(image_prefix, path_transform(image_path))

        transform: Optional
            albumentations transform class or None


        """

    def __init__(
            self,
            csv_path: str,
            image_prefix: str = '',
            path_transform: Callable = None,
            transform=None,
            return_label: bool = True,
            image_column: Optional[str] = None,
            target_column: Optional[str] = None,
    ):
        super().__init__(image_prefix=image_prefix, path_transform=path_transform, transform=transform)

        self.return_label = return_label
        dt = pd.read_csv(csv_path)
        self.images = dt.values[:, 0] if image_column is None else dt[image_column].values
        if self.return_label:
            self.targets = dt.values[:, 1] if target_column is None else dt[target_column].values

    def __len__(
            self
    ) -> int:
        """
        Return length of dataset

        Returns
        -------
        int
        """
        return len(self.images)

    def __getitem__(
            self,
            idx: int
    ) -> Dict:
        """
        Read object of dataset by index

        Attributes
        ----------
        idx: int
            index of object in dataset

        Returns
        -------
        Dict
            {"image": ...} or {"image": ..., "annotations": ...}
        """
        image = self._read_image(self.images[idx])
        if not self.return_label:
            return {
                'image': image
            }
        label = json.loads(self.targets[idx])
        classes = np.array([_['class'] for _ in label], dtype='int32')
        boxes = np.stack([
            np.array([_['x1'], _['y1'], _['x2'], _['y2']], dtype='int32')
            for _ in label
        ])
        annotations = np.concatenate((boxes, classes), axis=1)
        if self.transform:
            result = self.transform(image=image, bboxes=annotations)
            image = result['image']
            annotations = result['bboxes']
        return {
            'image': image,
            'annotations': annotations
        }

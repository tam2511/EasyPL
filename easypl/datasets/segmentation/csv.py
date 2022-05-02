from typing import Callable, Optional, Union, List, Dict

import cv2
import pandas as pd

from easypl.datasets.base import PathBaseDataset


class CSVDatasetSegmentation(PathBaseDataset):
    """
    Csv dataset for segmentation

    Attributes
    ----------
    csv_path: str
        path to csv file with paths of images

    return_label: bool
        if True return (image, label), else return only image

    image_column: Optional[str]
        column name or None. If None then will be getting the first column

    target_column: Optional[str]
        column name or None. If None then will be getting all but the first column

    image_prefix: str
        path prefix which will be added to paths of images in csv file

    mask_prefix: str
        path prefix which will be added to paths of masks in csv file

    path_transform: Optional[Callable]
        None or function for transform of path. Will be os.path.join(image_prefix, path_transform(image_path))

    transform: Optional
        albumentations transform class or None


    """

    def __init__(
            self,
            csv_path: str,
            image_prefix: str = '',
            mask_prefix: str = '',
            path_transform: Optional[Callable] = None,
            transform: Optional = None,
            return_label: bool = True,
            image_column: Optional[str] = None,
            target_column: Optional[str] = None,
    ):
        super().__init__(image_prefix=image_prefix, path_transform=path_transform, transform=transform)
        self.return_label = return_label
        self.mask_prefix = mask_prefix
        dt = pd.read_csv(csv_path)
        self.images = dt.values[:, 0] if image_column is None else dt[image_column].values
        self.masks = self.images if target_column is None else dt[target_column].values

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
            {"image": ...} or {"image": ..., "mask": ...}
        """
        image = self._read_image(self.images[idx])
        if not self.return_label:
            return {
                'image': image
            }
        mask = self._read_image(
            self.masks[idx],
            image_prefix=self.mask_prefix,
            read_flag=cv2.IMREAD_GRAYSCALE,
            to_rgb=False
        ).astype('int64')
        return {
            'image': image,
            'mask': mask
        }

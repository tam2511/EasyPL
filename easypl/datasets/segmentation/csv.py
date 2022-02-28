from typing import Callable, Optional, Union, List

import cv2
import pandas as pd

from easypl.datasets.base import PathBaseDataset


class CSVDatasetSegmentation(PathBaseDataset):
    '''
    Csv dataset representation
    '''

    def __init__(
            self,
            csv_path: str,
            image_prefix: str = '',
            mask_prefix: str = '',
            path_transform: Callable = None,
            transform=None,
            return_label: bool = True,
            image_column: Optional[str] = None,
            target_column: Optional[str] = None,
    ):
        '''
        :param csv_path: path to csv file with paths of images (one column)
        :param image_prefix: path prefix which will be added to paths of images in csv file
        :param mask_prefix: path prefix which will be added to paths of masks in csv file
        :param path_transform: None or function for transform of path. Will be os.path.join(image_prefix,
         path_transform(image_path))
        :param transform: albumentations transform class or None
        :param return_label: if True return (image, label), else return only image
        :param image_column: column name or None. If None then will be getting the first column
        :param target_column: column name or None. If None then will be getting image paths
        '''
        super().__init__(image_prefix=image_prefix, path_transform=path_transform, transform=transform)
        self.return_label = return_label
        self.mask_prefix = mask_prefix
        dt = pd.read_csv(csv_path)
        self.images = dt.values[:, 0] if image_column is None else dt[image_column].values
        self.masks = self.images if target_column is None else dt[target_column].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> dict:
        image = self._read_image(self.images[idx])
        if not self.return_label:
            return {
                'image': image
            }
        mask = self._read_image(
            self.masks[idx], image_prefix=self.mask_prefix, read_flag=cv2.IMREAD_GRAYSCALE, to_rgb=False
        ).astype('int64')
        return {
            'image': image,
            'mask': mask
        }

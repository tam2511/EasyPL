from typing import Callable, Optional, Union, List
import pandas as pd

from easypl.datasets.base import PathBaseDataset


class CSVDatasetClassification(PathBaseDataset):
    '''
    Csv dataset representation
    '''

    def __init__(
            self,
            csv_path: str,
            image_prefix: str = '',
            path_transform: Callable = None,
            transform=None,
            return_label: bool = True,
            image_column: Optional[str] = None,
            target_columns: Optional[Union[str, List[str]]] = None,
    ):
        '''
        :param csv_path: path to csv file with paths of images (one column)
        :param image_prefix: path prefix which will be added to paths of images in csv file
        :param path_transform: None or function for transform of path. Will be os.path.join(image_prefix,
         path_transform(image_path))
        :param transform: albumentations transform class or None
        :param return_label: if True return (image, label), else return only image
        :param image_column: column name or None. If None then will be getting the first column
        :param target_columns: column name/names or None. If None then will be getting all but the first column
        '''
        super().__init__(image_prefix=image_prefix, path_transform=path_transform, transform=transform)
        self.return_label = return_label
        dt = pd.read_csv(csv_path)
        self.images = dt.values[:, 0] if image_column is None else dt[image_column].values
        self.targets = dt.values[:, 1:] if target_columns is None else dt[target_columns].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> dict:
        image = self._read_image(self.images[idx])
        if not self.return_label:
            return {
                'image': image
            }
        label = self.targets[idx].astype('int64')
        return {
            'image': image,
            'target': label
        }

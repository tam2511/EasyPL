from typing import Callable
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
            return_label: bool = True
    ):
        '''
        :param csv_path: path to csv file with paths of images (one column)
        :param image_prefix: path prefix which will be added to paths of images in csv file
        :param path_transform: None or function for transform of path. Will be os.path.join(image_prefix,
         path_transform(image_path))
        :param transform: albumentations transform class or None
        :param return_label: if True return (image, label), else return only image
        '''
        super().__init__(image_prefix=image_prefix, path_transform=path_transform, transform=transform)
        self.return_label = return_label
        self.dt = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.dt)

    def __getitem__(self, idx) -> dict:
        row = self.dt.iloc[idx].values
        image = self._read_image(row[0])
        if not self.return_label:
            return {
                'image': image
            }
        label = row[1:].astype('int64')
        return {
            'image': image,
            'target': label
        }

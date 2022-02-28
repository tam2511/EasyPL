from typing import Callable
import os
from torch.utils.data import Dataset
import dill

from easypl.datasets.utils import read_image


class PathBaseDataset(Dataset):
    '''
    Abstract class of path based dataset
    '''

    def __init__(
            self,
            image_prefix: str = '',
            path_transform: Callable = None,
            transform=None
    ):
        '''
        :param image_prefix: path prefix which will be added to paths of images in csv file
        :param path_transform: None or function for transform of path. Will be os.path.join(image_prefix,
         path_transform(image_path))
        :param transform: albumentations transform class or None
        '''
        self.image_prefix = image_prefix
        self.transform = transform
        self.path_transform = dill.dumps(path_transform, recurse=True) if path_transform is not None else None

    def __len__(self):
        raise NotImplementedError

    def _read_image(self, image_id, image_prefix=None, **image_read_kwargs):
        image_path = image_id
        if self.path_transform is not None:
            image_path = dill.loads(self.path_transform)(image_path)
        if image_prefix is not None:
            image_path = os.path.join(image_prefix, image_path)
        elif self.image_prefix != '':
            image_path = os.path.join(self.image_prefix, image_path)
        image = read_image(image_path, **image_read_kwargs)
        if self.transform:
            image = self.transform(image=image)['image']
        return image

    def __getitem__(self, idx) -> dict:
        raise

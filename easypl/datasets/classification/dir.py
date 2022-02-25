import os
from typing import Callable
import dill

from easypl.datasets.base import PathBaseDataset


class DirDatasetClassification(PathBaseDataset):
    '''
    Dataset implementation for images in directory on disk (stored images paths in RAM).
    Require root_path/.../image_path structure.
    '''

    def __init__(
            self,
            root_path: str,
            label_parser: Callable,
            transform=None,
            return_label: bool = True
    ):
        '''
        :param root_path: path of directory with images
        :param transform: albumentations transform or None
        :param return_label: if True return (image, label), else return only image
        :param label_parser: function for parsing label from relative path
        '''
        super().__init__(image_prefix=root_path, transform=transform)
        self.return_label = return_label
        self.label_parser = label_parser

        self.image_paths = []
        self.labels = []
        self.__load(root_path)

    def __get_label(self, path):
        if self.label_parser is not None:
            return dill.loads(self.label_parser)(path)

    def __load(self, root_path):
        for root, _, files in os.walk(root_path):
            for file_name in files:
                self.image_paths.append(file_name)
                if self.return_label:
                    self.labels.append(self.__get_label(file_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> dict:
        image_path = self.image_paths[idx]
        image = self._read_image(image_path)
        if not self.return_label:
            return {
                'image': image
            }
        label = self.labels[idx]
        return {
            'image': image,
            'target': label
        }

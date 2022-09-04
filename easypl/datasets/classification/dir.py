import os
from typing import Callable, Optional, Dict
import dill

from easypl.datasets.base import PathBaseDataset


class DirDatasetClassification(PathBaseDataset):
    """
    Dataset implementation for images in directory on disk (stored images paths in RAM).
    Require root_path/.../image_path structure.

    Attributes
    ----------
    root_path: str
        path of directory with images

    transform: Optional
        albumentations transform or None

    return_label: bool
        if True return dict with two keys (image, target), else return dict with one key (image)

    label_parser: Callable
        function for parsing label from relative path

    """

    def __init__(
            self,
            root_path: str,
            label_parser: Callable,
            transform: Optional = None,
            return_label: bool = True
    ):
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

    def __len__(
            self
    ) -> int:
        """
        Return length of dataset

        Returns
        -------
        int
        """
        return len(self.image_paths)

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
            {"image": ...} or {"image": ..., "target": ...}
        """
        image_path = self.image_paths[idx]
        image = self._read_image(image_path)
        if self.transform:
            image = self.transform(image=image)['image']
        if not self.return_label:
            return {
                'image': image
            }
        label = self.labels[idx]
        return {
            'image': image,
            'target': label
        }

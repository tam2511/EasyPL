from typing import Callable, Optional, Any, Dict
import os
from torch.utils.data import Dataset
import dill

from easypl.datasets.utils import read_image


class PathBaseDataset(Dataset):
    """
    Abstract class of path based dataset

    Attributes
    ----------
    image_prefix: str
        path prefix which will be added to paths of images in csv file

    path_transform: Optional[Callable]
        None or function for transform of path. Will be os.path.join(image_prefix, path_transform(image_path))

    transform: Optional
        albumentations transform class or None


    """

    def __init__(
            self,
            image_prefix: str = '',
            path_transform: Optional[Callable] = None,
            transform: Optional = None
    ):
        self.image_prefix = image_prefix
        self.transform = transform
        self.path_transform = dill.dumps(path_transform, recurse=True) if path_transform is not None else None

    def __len__(
            self
    ) -> int:
        """
        Return length of dataset

        Returns
        -------
        int
        """
        raise NotImplementedError

    def _read_image(
            self,
            image_id: Any,
            image_prefix: Optional[str] = None,
            **image_read_kwargs
    ) -> Any:
        """
        Read image from disk

        Attributes
        ----------
        image_id: Any
            Any image identifier

        image_prefix: Optional
            prefix identifier with os.path.join if is str

        image_read_kwargs
            Additional arguments for function  `easypl.datasets.utils.read_image`

        Returns
        -------
        Any
            image object
        """
        image_path = image_id
        if self.path_transform is not None:
            image_path = dill.loads(self.path_transform)(image_path)
        if image_prefix is not None:
            image_path = os.path.join(image_prefix, image_path)
        elif self.image_prefix != '':
            image_path = os.path.join(self.image_prefix, image_path)
        image = read_image(image_path, **image_read_kwargs)
        return image

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
            object of dataset if dict format
        """
        raise NotImplementedError

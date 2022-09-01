from abc import abstractmethod
from typing import Any

import torch


class BasePostprocessing(object):

    @abstractmethod
    def __call__(
            self,
            inputs: Any,
            outputs: Any
    ) -> torch.Tensor:
        raise NotImplementedError

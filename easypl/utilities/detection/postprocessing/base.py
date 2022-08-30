from abc import abstractmethod
from typing import Any

import torch


class BasePostprocessing(object):
    @abstractmethod
    def targets_handle(
            self,
            targets: Any
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def outputs_handle(
            self,
            outputs: Any
    ) -> torch.Tensor:
        raise NotImplementedError

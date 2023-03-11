from abc import abstractmethod
from typing import Optional, Any

import torch


class BasePostprocessing(object):
    def targets_handle(
            self,
            annotations: torch.Tensor,
            image_infos: Optional = None
    ) -> torch.Tensor:
        """
        Target handling for predict stage. Must return annotations in (x1, y1, x2, y2, class_idx) form.
        By default, it returns annotations without transforms.

        Attributes
        ----------
        annotations: torch.Tensor
            Targets with size [batch size X max annotation size X 5].

        image_infos: Optional
            Images infos

        Returns
        ----------
        torch.Tensor
            Transformed annotations

        Examples
        ----------
            >>> def targets_handle(
            ...        self,
            ...        annotations: torch.Tensor,
            ...        image_infos: Dict[str, torch.Tensor]
            ...) -> torch.Tensor:
            ...    annotations_ = annotations.detach().clone()
            ...    annotations_[:, :, :4] *= image_infos['scale']
            ...    return annotations_
        """
        return annotations

    @abstractmethod
    def outputs_handle(
            self,
            outputs: Any,
            image_infos: Optional = None
    ) -> torch.Tensor:
        """
        Outputs handling for predict stage. Must return result in (x1, y1, x2, y2, class_prob, class_idx) form.

        Attributes
        ----------
        outputs: Any
            Outputs from model.

        image_sizes: torch.Tensor
            Sizes of images [batch size X 2]

        image_infos: Optional
            Images infos

        Returns
        ----------
        torch.Tensor
            Transformed model outputs.

        """
        raise NotImplementedError

from abc import abstractmethod
from typing import Any

import torch


class BasePostprocessing(object):
    def targets_handle(
            self,
            annotations: torch.Tensor,
            image_scales: torch.Tensor
    ) -> torch.Tensor:
        """
        Target handling for predict stage. Must return annotations in (x1, y1, x2, y2, class_idx) form.
        By default it returns annotations without transforms.

        Attributes
        ----------
        annotations: torch.Tensor
            Targets with size [batch size X max annotation size X 5].

        image_scales: torch.Tensor
            Scales of images [batch size X 2]

        Returns
        ----------
        torch.Tensor
            Transformed annotations

        Examples
        ----------
            >>> def targets_handle(
            ...        self,
            ...        annotations: torch.Tensor,
            ...        image_scales: torch.Tensor
            ...) -> torch.Tensor:
            ...    annotations_ = annotations.detach().clone()
            ...    annotations_[:, :, :4] *= image_scales[:, 0].repeat(1, 2).unsqueeze(1).repeat(1, annotations_.shape[1], 1)
            ...    return annotations_
        """
        return annotations

    @abstractmethod
    def outputs_handle(
            self,
            outputs: Any,
            image_sizes: torch.Tensor,
            image_scales: torch.Tensor,
    ) -> torch.Tensor:
        """
        Outputs handling for predict stage. Must return result in (x1, y1, x2, y2, class_idx, class_prob) form.

        Attributes
        ----------
        outputs: Any
            Outputs from model.

        image_sizes: torch.Tensor
            Sizes of images [batch size X 2]

        image_scales: torch.Tensor
            Scales of images [batch size X 2]

        Returns
        ----------
        torch.Tensor
            Transformed model outputs.

        """
        raise NotImplementedError

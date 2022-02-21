from typing import Optional, Union

import torch


def to_(
        obj,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None
):
    '''
    Passing data any structure to device and dtype
    :param obj: data
    :param device: device string or torch device
    :param dtype: torch dtype
    :return: passied copied data
    '''
    if isinstance(obj, torch.Tensor):
        return obj.to(device=obj.device if device is None else device, dtype=obj.dtype if dtype is None else dtype)
    elif isinstance(obj, dict):
        return {k: to_(obj[k], device) for k in obj}
    elif isinstance(obj, list):
        return [to_(o, device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple([to_(o, device) for o in obj])
    else:
        return obj


def multiple_unsqueeze(
        tensor: torch.Tensor,
        ndims: int = 0,
        dim: int = 0
):
    '''
    Apply torch.unsqueeze ndims times belong dim
    :param tensor: target tensor
    :param ndims: number times for unsqueeze
    :param dim: target dimension
    '''
    for _ in range(ndims):
        tensor = tensor.unsqueeze(dim)
    return tensor

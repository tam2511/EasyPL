from typing import Optional, Union, List
import numpy as np
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
) -> torch.Tensor:
    '''
    Apply torch.unsqueeze ndims times belong dim
    :param tensor: target tensor
    :param ndims: number times for unsqueeze
    :param dim: target dimension
    '''
    for _ in range(ndims):
        tensor = tensor.unsqueeze(dim)
    return tensor


def optimal_grid_size(
        n: int
) -> List:
    '''
    Get optimal grid size
    :param n: number of tiles
    :return: grid size that contains defined number of tiles "n"
    '''
    x1 = np.ceil((np.sqrt(4 * n + 1) + 1) / 2)
    y1 = x1 - 1
    x2 = np.floor((np.sqrt(4 * n + 1) + 1) / 2)
    y2 = np.ceil(n / x2)
    return [int(x1), int(y1)] if x1 * y1 <= x2 * y2 else [int(x2), int(y2)]


def grids(
        width,
        height,
        n
) -> List:
    '''
    Get grids for matrix that contains defined number of tiles
    :param width: width of matrix
    :param height: height of matrix
    :param n: number of tiles
    :return: list of grid coordinates
    '''
    n_w_grid, n_h_grid = sorted(optimal_grid_size(n), reverse=width >= height)
    w_grid, h_grid = int(np.ceil(width / n_w_grid)), int(np.ceil(height / n_h_grid))
    w_grids, h_grids = list(range(0, width, w_grid)) + [width], list(range(0, height, h_grid)) + [height]
    grids_ = []
    for w_idx in range(0, len(w_grids) - 1):
        for h_idx in range(0, len(h_grids) - 1):
            grids_.append(((w_grids[w_idx], w_grids[w_idx + 1]), (h_grids[h_idx], h_grids[h_idx + 1])))
    return grids_

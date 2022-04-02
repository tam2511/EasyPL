from torchmetrics import Metric
from typing import Union, Callable
import torch
from torch.nn.functional import normalize

from easypl.metrics.utils import build_distance, available_distances


class SearchAccuracy(Metric):
    """
    Version of accuracy for search case
    """

    def __init__(
            self,
            k: Union[int, list] = 1,
            batch_size: int = 512,
            distance: Union[str, Callable] = 'L2',
            largest: bool = True,
            dist_sync_on_step: bool = False,
            compute_on_step: bool = True
    ):
        """
        :param k: SearchAccuracy return top k (top (k[0], k[1], ...) if k is list) accuracy rate
        :param batch_size: batch size for evaluate distance operations
        :param distance: name or function of distance
        :param largest: if True metric evaluate top largest samples, else evaluate smallest samples
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.k = k
        self.batch_size = batch_size
        if isinstance(distance, str):
            self.distance = build_distance(distance)
            self.distance_name = distance
        else:
            self.distance = distance
            self.distance_name = 'User\' function'
        self.__check_distance()
        self.largest = largest
        self.name = self.__class__.__name__
        self.add_state('embeddings', default=[], dist_reduce_fx='cat')
        self.add_state('targets', default=[], dist_reduce_fx='cat')

    def __check_distance(self):
        try:
            result = self.distance(torch.rand(2, 32), torch.rand(3, 32))
            assert result.shape == (2, 3)
        except Exception:
            raise ValueError('Distance function is incorrect.')

    def __compute(self, embeddings: torch.Tensor, targets: torch.Tensor):
        k = [self.k] if isinstance(self.k, int) else list(self.k)
        max_k = min(max(k) + 1, embeddings.size(0))
        result = [0.0] * len(k)
        if self.distance_name == 'Normalize cosine':
            embeddings = normalize(embeddings, dim=1)
        for idx in range(0, embeddings.size(0), self.batch_size):
            query = embeddings.narrow(0, idx, min(self.batch_size, embeddings.size(0) - idx))
            pairwaise_matrix = self.distance(query, embeddings)
            _, indicies = torch.topk(pairwaise_matrix, k=max_k, dim=1, largest=self.largest)
            true_targets = targets.narrow(0, idx, min(self.batch_size, embeddings.size(0) - idx))
            predicted_targets = targets[indicies]
            tp = true_targets.unsqueeze(-1).repeat(1, predicted_targets.size(1)) == predicted_targets
            tp[:, 0] = False
            for k_idx in range(len(k)):
                result[k_idx] += tp.narrow(1, 0, min(k[k_idx] + 1, tp.size(1))).any(dim=1).sum()
        result = torch.tensor(result) / embeddings.size(0)
        return {
            '{}_top{}'.format(self.name, k[k_idx]): result[[k_idx]] for k_idx in range(len(k))
        }

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.embeddings.append(preds)
        self.targets.append(targets)

    def compute(self):
        embeddings = torch.cat(self.embeddings, dim=0) if isinstance(self.embeddings, list) else self.embeddings
        self.embeddings = []
        targets = torch.cat(self.targets, dim=0) if isinstance(self.targets, list) else self.targets
        self.targets = []
        return self.__compute(embeddings, targets)

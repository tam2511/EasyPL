from random import choices
import torch


class ImageCollector(object):
    '''
    Data collector with different strategies.
    '''

    def __init__(
            self,
            mode='first',
            max_images=1,
            score_func=None,
            largest=True,
            dataset_size=None
    ):
        '''
        :param mode: how will be collected results, availbale: ["first", "random", "top"]
        :param max_images: number of images which will be logged
        :param score_func: function for score evaluation, has input with two arguments "output" and "target"
        :param largest: if True will choose samples with higher score, otherwise with low score
        :param dataset_size: size of dataset
        '''
        self.mode = mode
        self.max_images = max_images
        self.score_func = score_func
        self.largest = largest
        self.dataset_size = dataset_size

        self.idx = 0
        self.idxs = None
        self.cache = {}
        self.reset()

    def update(self, output: torch.Tensor, target: torch.Tensor, data=None):
        if self.mode in ['random', 'first']:
            available_idxs = torch.where(torch.logical_not(torch.isinf(self.cache['score'])))[0]
            if self.idx in self.idxs and len(available_idxs) > 0:
                idx = available_idxs[0]
                self.idx += 1
                self.cache['score'][idx] = 0
                self.cache['data'][idx] = {
                    'output': output,
                    'target': target,
                    'data': data,
                    'score': 0
                }
            else:
                self.idx += 1
        else:
            score = self.score_func(output, target)
            dist = (self.cache['score'] - score) * (1 - 2 * self.largest)
            idx = torch.argmax(dist)
            if dist[idx] > 0:
                self.cache['score'][idx] = score
                self.cache['data'][idx] = {
                    'output': output,
                    'target': target,
                    'data': data,
                }
            self.idx += 1

    def compute(self) -> list:
        idxs = torch.where(torch.logical_not(torch.isinf(self.cache['score'])))[0]
        return [
            {
                'output': self.cache['data'][idx]['output'],
                'target': self.cache['data'][idx]['target'],
                'data': self.cache['data'][idx]['data'],
            }
            for idx in idxs
        ]

    def reset(self):
        self.idx = 0
        max_images = self.max_images
        if self.dataset_size is not None:
            max_images = min(max_images, self.dataset_size)
        if self.mode == 'first':
            self.idxs = list(range(self.max_images))
        elif self.mode == 'random':
            if self.dataset_size is None:
                raise ValueError('In "random" mode you should define "dataset_size" argument.')
            self.idxs = choices(list(range(self.dataset_size)), k=max_images)
        elif self.mode == 'top':
            if self.score_func is None:
                raise ValueError('In "top" mode you should define "score_func" argument.')
        self.cache['score'] = torch.ones(max_images) * (-float('inf') if self.largest else float('inf'))
        self.cache['data'] = [{} for _ in range(max_images)]

from torch.optim import Optimizer


class WrapperOptimizer(object):
    '''
    Wrapper for pytorch Optimizer class.
    '''

    def __init__(
            self,
            optimizer_cls,
            **kwargs
    ):
        '''
        :param optimizer_cls: pytorch Optimizer class
        '''
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs

    def __call__(
            self,
            params
    ) -> Optimizer:
        return self.optimizer_cls(params=params, **self.kwargs)

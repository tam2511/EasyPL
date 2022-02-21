from torch.optim import Optimizer
from inspect import signature


class WrapperScheduler(object):
    '''
    Wrapper for pytorch Learning rate Scheduler class.
    '''

    def __init__(
            self,
            scheduler_class,
            **kwargs
    ):
        '''
        :param scheduler_class: pytorch Learning rate Scheduler class
        '''
        self.scheduler_class = scheduler_class
        self.kwargs = {arg: kwargs[arg] for arg in kwargs if arg in dict(signature(scheduler_class).parameters)}
        self.options = {arg: kwargs[arg] for arg in kwargs if arg not in self.kwargs}

    def __call__(
            self,
            optimizer: Optimizer
    ):
        '''
        :param optimizer:  WrapperOptimizer object
        :return: Options of Learning rate Scheduler
        '''
        scheduler = self.scheduler_class(optimizer=optimizer, **self.kwargs)
        self.options['scheduler'] = scheduler
        return self.options

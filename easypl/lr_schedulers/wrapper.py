from torch.optim import Optimizer
from inspect import signature


class WrapperScheduler(object):
    """
    Wrapper for pytorch Learning rate Scheduler class.

    Attributes
    ----------
    scheduler_class
        Pytorch Learning rate Scheduler class.

    """

    def __init__(
            self,
            scheduler_class,
            **kwargs
    ):
        self.scheduler_class = scheduler_class
        self.kwargs = {arg: kwargs[arg] for arg in kwargs if arg in dict(signature(scheduler_class).parameters)}
        self.options = {arg: kwargs[arg] for arg in kwargs if arg not in self.kwargs}

    def __call__(
            self,
            optimizer: Optimizer
    ):
        """
        Return lr_scheduler

        Attributes
        ----------
        optimizer: Optimizer
            WrapperOptimizer object

        """
        scheduler = self.scheduler_class(optimizer=optimizer, **self.kwargs)
        options = self.options.copy()
        options['scheduler'] = scheduler
        return options

from torch.optim import Optimizer


class WrapperOptimizer(object):
    """
    Wrapper for pytorch Optimizer class.

    Attributes
    ----------------
    optimizer_cls
        Pytorch Optimizer class.

    kwargs
        Additional arguments.
    """

    def __init__(
            self,
            optimizer_cls,
            **kwargs
    ):
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs

    def __call__(
            self,
            params
    ) -> Optimizer:
        """
        Return optimizer.

        Attributes
        ----------
        params
            Model params

        Returns
        ----------
        Optimizer
            Optimizer object
        """
        return self.optimizer_cls(params=params, **self.kwargs)

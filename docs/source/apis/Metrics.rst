Metrics
==================================

.. toctree::
   :maxdepth: 3

   metrics/Classification
   metrics/Segmentation
   metrics/Detection

.. autoclass:: easypl.metrics.base.MetricsList
    :members:
    :exclude-members: compute, reset, update

.. autoclass:: easypl.metrics.torch.TorchMetric
    :members:
    :exclude-members: compute, reset, update

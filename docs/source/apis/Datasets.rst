Datasets
==================================

For EasyPL to work correctly, your dataset must return a dict.
For ease of creating such a class, we prepared the base class PathBaseDataset.

.. autoclass:: easypl.datasets.base.PathBaseDataset
    :members:

    .. automethod:: easypl.datasets.base.PathBaseDataset.__len__
    .. automethod:: easypl.datasets.base.PathBaseDataset.__getitem__
    .. automethod:: easypl.datasets.base.PathBaseDataset._read_image

For correctly using `PathBaseDataset` you should override `__len__` and `__getitem__` methods.
You can use `_read_image` method for simply image loading.

We create simply examples of datasets for classification and segmentation tasks. See below.

.. toctree::
   :maxdepth: 3

   datasets/classification
   datasets/segmentation

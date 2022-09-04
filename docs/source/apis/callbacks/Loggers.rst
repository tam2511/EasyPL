Loggers
==================================


.. toctree::
   :maxdepth: 3

   loggers/BaseImageLogger
   loggers/ClassificationImageLogger
   loggers/SegmentationImageLogger
   loggers/DetectionImageLogger

.. autoclass:: easypl.callbacks.loggers.base.BaseSampleLogger
    :members: _log_wandb, _log_tensorboard, _log_on_disk, _post_init
    :exclude-members: on_predict_batch_end, on_predict_epoch_end, on_predict_start, on_test_batch_end, on_test_epoch_end, on_test_start, on_train_batch_end, on_train_epoch_end, on_train_start, on_validation_batch_end, on_validation_epoch_end, on_validation_start


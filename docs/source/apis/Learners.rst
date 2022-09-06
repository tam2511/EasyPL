Learners
==================================

.. toctree::
   :maxdepth: 3

   learners/ClassificationLearner
   learners/RecognitionLearner
   learners/SegmentationLearner
   learners/DetectionLearner
   learners/GANLearner


.. autoclass:: easypl.learners.base.BaseLearner
    :members:
    :exclude-members: configure_optimizers, test_epoch_end, test_step, training_epoch_end, training_step, validation_epoch_end, validation_step
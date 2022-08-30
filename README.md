# EasyPL - set of wrappers and tools based on PyTorch Lightning to quickly start learning Pytorch models.

This library is a template project for faster deployment of machine learning model training projects based on PyTorch Lightning. If PyTorch Lightning makes learning models easy, then EasyPL makes it super easy.

## Quickstart

You can install this library using pip:

```
pip install easyplib
```
Note: Sorry for the mismatch between the library name in the pypi index and the documentation. The pypi project name normalization algorithms does not allow you to specify an easypl project name.

Also you can install library manually:

```
git clone https://github.com/tam2511/EasyPL.git
cd EasyPL
python setup.py install
```

You can find a description of all functions and API in the [documentation](https://easypl.readthedocs.io/en/latest/index.html).

## Examples

You can find all examples on [rtd](https://easypl.readthedocs.io/en/latest/Examples.html) with full training pipelines.

For the library to work correctly, you need to wrap your optimizer and lr scheduler in the appropriate classes, for example:

```python
from easypl.optimizers import WrapperOptimizer
from easypl.lr_schedulers import WrapperScheduler

optimizer = WrapperOptimizer(optim.Adam, lr=1e-4)
lr_scheduler = WrapperScheduler(optim.lr_scheduler.StepLR, step_size=2, gamma=1e-1, interval='epoch')
```

When using metrics from the torchmetrics library, you can use the TorchMetric wrapper:

```python
from easypl.metrics import TorchMetric

TorchMetric(F1(num_classes=2, average='none'), class_names=['cat', 'dog'])
```

There are many callbacks available in the EasyPL library. For example, callbacks for image logging, cutmix and test-time augmentation are defined below.

```python
image_logger = ClassificationImageLogger(
    phase='train',
    max_samples=10,
    class_names=['cat', 'dog'],
    max_log_classes=2,
    dir_path='images',
    save_on_disk=True,
)

# Cutmix callback
cutmix = Cutmix(
    on_batch=True,
    p=1.0,
    domen='classification',
)

# Test time augmentation callback
tta = ClassificationImageTestTimeAugmentation(
    n=2,
    augmentations=[VerticalFlip(p=1.0)],
    phase='val'
)
```

The final part of the training pipeline is the definition of the Learner class and the standard launch of training through the Trainer from the PyTorch Lightning library.

```python
learner = ClassificatorLearner(
    model=model,
    loss=loss_f,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    train_metrics=train_metrics,
    val_metrics=val_metrics,
    data_keys=['image'],
    target_keys=['target'],
    multilabel=False
)
trainer = Trainer(
    gpus=1,
    callbacks=[image_logger, cutmix, tta],
    max_epochs=3,
    precision=16
)
trainer.fit(learner, train_dataloaders=train_dataloader, val_dataloaders=[val_dataloader])

```

## TODO
- [ ] Learner for image detection task.
- [ ] Learner for regression task.
- [ ] Example learner for GAN training.
- [ ] Callbacks for target/sample analytics.
- [ ] Finish writing detection part of callbacks.
- [ ] Add tests.

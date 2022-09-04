Simple multiclass classification example
===========================

We can observe simple example for classification task with two classes (cats and dogs).


First, you should import common libraries and packages below. (If you don't have some package, than install it).

.. code-block:: python

    import cv2
    from torch.utils.data import Dataset, DataLoader
    import torch
    import torch.optim as optim
    import torch.nn as nn
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from albumentations.augmentations import *
    from albumentations.core.composition import *
    from albumentations.pytorch.transforms import *
    from timm import create_model
    import random
    from torchmetrics import *
    import shutil


Than you can import EasyPL packages, as like:

.. code-block:: python

    from easypl.learners import ClassificationLearner
    from easypl.metrics import TorchMetric
    from easypl.optimizers import WrapperOptimizer
    from easypl.lr_schedulers import WrapperScheduler
    from easypl.datasets import CSVDatasetClassification
    from easypl.callbacks import ClassificationImageLogger
    from easypl.callbacks import Cutmix
    from easypl.callbacks import ClassificationImageTestTimeAugmentation


Than you should define datasets and dataloaders. You can use this simple example:

.. code-block:: python

    train_transform = Compose([
        HorizontalFlip(p=0.5),
        Rotate(p=0.5),
        LongestMaxSize(max_size=224),
        PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        Normalize(),
        ToTensorV2(),
    ])

    val_transform = Compose([
        LongestMaxSize(max_size=600),
        PadIfNeeded(min_height=600, min_width=600, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        Normalize(),
        ToTensorV2(),
    ])

    train_dataset = CSVDatasetClassification('../input/cat-dog-test/train.csv', image_prefix='../input/cat-dog-test/train', transform=train_transform, return_label=True)
    val_dataset = CSVDatasetClassification('../input/cat-dog-test/val.csv', image_prefix='../input/cat-dog-test/val', transform=val_transform, return_label=True)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=2)
    val_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=2)

Than we should define model (used timm), loss function, optimizer and metrics:

.. code-block:: python

    model = create_model('resnet18', pretrained=True, num_classes=2)

    loss_f = nn.CrossEntropyLoss()

    optimizer = WrapperOptimizer(optim.Adam, lr=1e-4)
    lr_scheduler = WrapperScheduler(optim.lr_scheduler.StepLR, step_size=2, gamma=1e-1, interval='epoch')

    train_metrics = []
    val_metrics = [TorchMetric(F1(num_classes=2, average='none'), class_names=['cat', 'dog'])]

If you need in callbacks, you can use our simple realization. Creating of callbacks looks like:

.. code-block:: python

    # Logger of outputs (images)
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

In finally, we should define learner and trainer, and than run training.

.. code-block:: python

    learner = ClassificationLearner(
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

Simple semantic segmentation example
===========================

We can observe simple example for segmentation task with one class (text).


First, you should import common libraries and packages below. (If you don't have some package, than install it).

.. code-block:: python

    import cv2
    import numpy as np
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

    from easypl.learners import SegmentationLearner
    from easypl.metrics import TorchMetric
    from easypl.metrics.segmentation import PixelLevelF1
    from easypl.optimizers import WrapperOptimizer
    from easypl.lr_schedulers import WrapperScheduler
    from easypl.datasets import CSVDatasetSegmentation
    from easypl.callbacks import SegmentationImageLogger
    from easypl.callbacks import Mixup
    from easypl.losses.segmentation import DiceLoss


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

    dataset = CSVDatasetSegmentation(
        csv_path='../input/lan-segmentation-1/train.csv',
        image_prefix='../input/lan-segmentation-1/train/images',
        mask_prefix='../input/lan-segmentation-1/train/masks',
        image_column='path',
        target_column='path'
    )

    class WrapperDataset(Dataset):
        def __init__(self, dataset, transform=None, idxs=[]):
            self.dataset = dataset
            self.transform = transform
            self.idxs = idxs

        def __getitem__(self, idx):
            idx = self.idxs[idx]
            row = self.dataset[idx]
            row['mask'][row['mask'] < 150] = 0
            row['mask'][row['mask'] > 150] = 1
            if self.transform:
                result = self.transform(image=row['image'], mask=row['mask'])
                row['image'] = result['image']
                row['mask'] = result['mask'].to(dtype=torch.long)
            row['mask'] = one_hot(row['mask'], num_classes=2).permute(2, 0, 1)
            return row

        def __len__(self):
            return len(self.idxs)

    image_size = 768

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.7),
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    size_dataset = len(dataset)
    val_size = int(size_dataset * 0.1)
    train_dataset = WrapperDataset(dataset, transform=train_transform, idxs=list(range(val_size, size_dataset)))
    val_dataset = WrapperDataset(dataset, transform=val_transform, idxs=list(range(val_size)))

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

Than we should define model (used timm), loss function, optimizer and metrics:

.. code-block:: python

    model = smp.UnetPlusPlus(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        decoder_use_batchnorm=False,
        classes=2,
    )

    loss_f = DiceLoss(weight=torch.tensor([1, 10]))

    optimizer = WrapperOptimizer(optim.Adam, lr=1e-4)

    num_epochs = 7
    num_gpus = 1

    lr_scheduler = WrapperScheduler(
        torch.optim.lr_scheduler.OneCycleLR, max_lr=3e-4, pct_start=1 / (num_epochs),
        total_steps=int(len(train_dataloader) * num_epochs / num_gpus) + 10, div_factor=1e+3, final_div_factor=1e+4,
        anneal_strategy='cos', interval='step'
    )

    class_names = ['background', 'text']


    train_metrics = [

    ]

    val_metrics = [
        TorchMetric(PixelLevelF1(average='none', num_classes=len(class_names)), class_names),
    ]

If you need in callbacks, you can use our simple realization. Creating of callbacks looks like:

.. code-block:: python

    # Logger of outputs (images)
    logger = SegmentationImageLogger(
        phase='val',
        max_samples=10,
        num_classes=2,
        save_on_disk=True,
        dir_path='images'
    )

    # Cutmix callback
    mixup = Mixup(
        on_batch=True,
        p=1.0,
        domen='segmentation',
    )

In finally, we should define learner and trainer, and than run training.

.. code-block:: python

    learner = SegmentationLearner(
        model=model,
        loss=loss_f,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        data_keys=['image'],
        target_keys=['mask'],
        multilabel=False
    )
    trainer = pl.Trainer(gpus=num_gpus, callbacks=[logger, mixup, checkpoint_callback], max_epochs=num_epochs)
    trainer.fit(learner, train_dataloaders=train_dataloader, val_dataloaders=[val_dataloader])

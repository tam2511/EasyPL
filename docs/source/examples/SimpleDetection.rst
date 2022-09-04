Simple detection train example
===========================

We can observe simple example for detection in pascal dataset using effdet project (https://github.com/rwightman/efficientdet-pytorch).


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

    from easypl.learners.detection import DetectionLearner
    from easypl.metrics.detection import FBetaDetection
    from easypl.optimizers import WrapperOptimizer
    from easypl.lr_schedulers import WrapperScheduler
    from easypl.datasets import CSVDatasetDetection
    from easypl.callbacks.loggers import DetectionImageLogger
    from easypl.callbacks.mixers import Mixup, Cutmix, Mosaic
    from easypl.callbacks.finetuners import OptimizerInitialization
    from easypl.utilities.detection import BasePostprocessing

Than you should define datasets and dataloaders. You can use this simple example:

.. code-block:: python

    train_transform = Compose([
        HorizontalFlip(p=0.5),
        LongestMaxSize(max_size=512),
        PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        Normalize(),
        ToTensorV2(),
    ], bbox_params=BboxParams(format='pascal_voc', min_visibility=0.1))

    val_transform = Compose([
        LongestMaxSize(max_size=512),
        PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        Normalize(),
        ToTensorV2(),
    ], bbox_params=BboxParams(format='pascal_voc', min_visibility=0.1))

    test_transform = Compose([
        Normalize(),
        ToTensorV2(),
    ], bbox_params=BboxParams(format='pascal_voc', min_visibility=0.1))


    def collate_fn(batch):
        images = torch.stack([_['image'] for _ in batch])
        max_anno_size = max(len(_['annotations'][0]) for _ in batch)
        image_sizes = torch.from_numpy(np.stack([_['image_size'] for _ in batch]))
        image_scales = torch.from_numpy(np.stack([_['image_scale'] for _ in batch]))
        annotations = torch.ones(len(batch), max_anno_size, 5, dtype=torch.float) * -1
        for i in range(len(batch)):
            annotations[i][:len(batch[i]['annotations'][0])] = batch[i]['annotations'][0]
        return {
            'image': images,
            'annotations': annotations,
            'image_size': image_sizes,
            'image_scale': image_scales
        }

    train_dataset = CSVDatasetDetection('../input/pascal/train.csv', image_prefix='../input/pascal/train', transform=train_transform, return_label=True)
    val_dataset = CSVDatasetDetection('../input/pascal/val.csv', image_prefix='../input/pascal/val', transform=val_transform, return_label=True)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=2)
    val_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=2)

Than we should define model (used effnet: https://github.com/rwightman/efficientdet-pytorch), loss function, optimizer and metrics:

.. code-block:: python

    from effdet import EfficientDet, get_efficientdet_config

    num_classes = 20

    config = get_efficientdet_config('tf_efficientdet_d0')

    model = EfficientDet(config, pretrained_backbone=True)
    model.reset_head(num_classes=num_classes)

    from effdet.anchors import Anchors, AnchorLabeler, generate_detections
    from effdet.loss import DetectionLoss

    class EfficientDetLoss(nn.Module):
        def __init__(self, model, create_labeler=False):
            super().__init__()
            self.model = model
            self.config = model.config  # FIXME remove this when we can use @property (torchscript limitation)
            self.num_levels = model.config.num_levels
            self.num_classes = model.config.num_classes
            self.anchors = Anchors.from_config(model.config)
            self.max_detection_points = model.config.max_detection_points
            self.max_det_per_image = model.config.max_det_per_image
            self.soft_nms = model.config.soft_nms
            self.anchor_labeler = AnchorLabeler(self.anchors, self.num_classes, match_threshold=0.5)
            self.loss_fn = DetectionLoss(model.config)

        def forward(self, x, target):
            class_out, box_out = x
            cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(target[:, :, :4], target[:, :, 4])
            loss, class_loss, box_loss = self.loss_fn(class_out, box_out, cls_targets, box_targets, num_positives)
            output = {'loss': loss, 'class_loss': class_loss, 'box_loss': box_loss}
            return output


    loss_f = EfficientDetLoss(model=model, create_labeler=True)

    num_epochs = 5
    num_gpus = 1

    optimizer = WrapperOptimizer(optim.Adam, lr=1e-4)


    lr_scheduler = WrapperScheduler(
        torch.optim.lr_scheduler.OneCycleLR, max_lr=3e-4, pct_start=1 / (num_epochs),
        total_steps=int(len(train_dataloader) * num_epochs / num_gpus) + 10, div_factor=1e+3, final_div_factor=1e+4,
        anneal_strategy='cos', interval='step'
    )

    train_metrics = []
    val_metrics = [FBetaDetection([0.5])]

If you need in callbacks, you can use our simple realization. Creating of callbacks looks like:

.. code-block:: python

    # Logger of outputs (images)
    image_logger = DetectionImageLogger(phase='val', num_classes=num_classes)

In finally, we should define learner and trainer, and than run training.

.. code-block:: python

    learner = DetectionLearner(
        model=model,
        loss=loss_f,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        data_keys=['image'],
        target_keys=['annotations'],
        postprocessing=EfficientdetPostprocessing(model),
        image_size_key='image_size',
        image_scale_key='image_scale'
    )
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        callbacks=[image_logger],
        max_epochs=num_epochs,
    #     precision=32
    )
    trainer.fit(learner, train_dataloaders=train_dataloader, val_dataloaders=[val_dataloader])

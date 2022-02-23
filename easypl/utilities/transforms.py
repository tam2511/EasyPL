from albumentations.augmentations.transforms import Normalize
from albumentations.core.composition import Compose
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import BasicTransform


class ToImage(BasicTransform):
    """
    Convert torch.Tensor to image in numpy format.
    """

    def __init__(self, transpose_mask=False, always_apply=True, p=1.0):
        super(ToImage, self).__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask, "masks": self.apply_to_masks}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        if len(img.shape) != 3:
            raise ValueError("Albumentations only supports tensors on CHW format")

        return img.permute(1, 2, 0).numpy()

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.permute(1, 2, 0)
        return mask.numpy()

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def get_transform_init_args_names(self):
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params):
        return {}


def inv_transform(transforms: list):
    inv_transforms = []
    for transform_idx in range(len(transforms)):
        transform_ = transforms[transform_idx]
        if isinstance(transform_, ToTensorV2):
            inv_transforms.append(
                ToImage(transpose_mask=transform_.transpose_mask, always_apply=transform_.always_apply, p=transform_.p))
        if isinstance(transform_, Normalize):
            inv_transforms.append(
                Normalize(mean=[- _ * transform_.max_pixel_value for _ in transform_.mean], std=[1, 1, 1],
                          max_pixel_value=1.0)
            )
            inv_transforms.append(
                Normalize(mean=[0, 0, 0], std=[1 / (_ * transform_.max_pixel_value) for _ in transform_.std],
                          max_pixel_value=1.0))
    inv_transforms = inv_transforms[::-1]
    return Compose(inv_transforms)

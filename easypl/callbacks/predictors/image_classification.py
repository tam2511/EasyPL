from typing import List, Dict, Union, Callable

from easypl.callbacks.predictors.base_image import BaseImageTestTimeAugmentation


class ClassificationImageTestTimeAugmentation(BaseImageTestTimeAugmentation):
    """ Image classification callback for test-time-augmentation """

    def __init__(
            self,
            n: int,
            augmentations: List,
            augmentation_method: str = 'first',
            phase='val',
            reduce_method: Union[str, Callable] = 'mean'
    ):
        """
        :param augmentations: list of augmentation transforms
        """
        super().__init__(n=n, augmentations=augmentations, augmentation_method=augmentation_method, phase=phase)
        self.reduce_method = reduce_method
        self.multilabel = None

    def post_init(self, trainer, pl_module):
        super().post_init(trainer, pl_module)
        self.multilabel = pl_module.multilabel
        if len(self.data_keys) != 1:
            raise ValueError(f'Size of "data_keys" must have len 1, but have len {len(self.data_keys)}')

    def metric_formatting(self, outputs, targets):
        outputs = outputs.sigmoid() if self.multilabel else outputs.argmax(dim=1)
        targets = targets if targets.ndim == 1 or self.multilabel else targets.argmax(dim=1)
        return outputs, targets

    def reduce(self, tensor):
        if isinstance(self.reduce_method, str):
            if self.reduce_method == 'mean':
                return tensor.mean(0)
            else:
                raise NotImplementedError
        else:
            return self.reduce_method(tensor)

    def augment(self, sample: Dict, augmentation) -> Dict:
        image = sample[self.data_keys[0]].copy()
        image = augmentation(image=image)['image']
        ret_sample = {key: sample[key] for key in sample if key not in self.data_keys}
        ret_sample[self.data_keys[0]] = image
        return ret_sample

    def preprocessing(self, sample: Dict, dataloader_idx: int) -> Dict:
        sample[self.data_keys[0]] = self.inv_transform[dataloader_idx](
            image=sample[self.data_keys[0]].cpu().numpy()
        )['image']
        return sample

    def postprocessing(self, sample: Dict, dataloader_idx: int) -> Dict:
        sample[self.data_keys[0]] = self.transform[dataloader_idx](image=sample[self.data_keys[0]])['image']
        return sample

import torch
import numpy as np
import pytorch_lightning
from pytorch_lightning.callbacks import Callback
from typing import List, Dict, Any, Tuple


class BaseTestTimeAugmentation(Callback):
    """
    Base callback for test-time-augmentation

    Attributes
    ----------
    n: int
        Number of augmentations.

    augmentations: List
        List of augmentations, which will be used.

    augmentation_method: str
        Method of selecting augmentations from list. Available: ["first", "random"]

    phase: str
        Phase which will be used by this predictor callback.
        Available: ["val", "test", "predict"].

    """

    def __init__(
            self,
            n: int,
            augmentations: List,
            augmentation_method: str = 'first',
            phase='val'
    ):
        super().__init__()
        self.n = n
        self.augmentations = augmentations
        self.augmentation_method = augmentation_method
        self.phase = phase

        if self.augmentation_method == 'first':
            self.current_n = min(self.n, len(self.augmentations))
        elif self.augmentation_method == 'random':
            self.current_n = self.n
        else:
            self.current_n = len(self.augmentations)
        self.data_keys = None
        self.collate_fns = []
        self.metrics = []

    def post_init(
            self,
            trainer: pytorch_lightning.Trainer,
            pl_module: pytorch_lightning.LightningModule
    ):
        """
        Abstract method for initialization in first batch handling. [NOT REQUIRED]

        Attributes
        ----------
        trainer: pytorch_lightning.Trainer
            Trainer of pytorch-lightning

        pl_module: pytorch_lightning.LightningModule
            LightningModule of pytorch-lightning

        """
        pass

    def on_phase_start(self, trainer, pl_module):
        if self.data_keys is None:
            pl_module.return_output_phase[self.phase] = True
            self.data_keys = pl_module.data_keys
            for dataloader_idx in range(len(trainer.__getattribute__(f'{self.phase}_dataloaders'))):
                self.collate_fns.append(
                    trainer.__getattribute__(f'{self.phase}_dataloaders')[dataloader_idx].collate_fn)
                trainer.__getattribute__(
                    f'{self.phase}_dataloaders'
                )[dataloader_idx].collate_fn = self.__collate_fn(dataloader_idx)
            if self.phase != 'predict':
                self.metrics = [pl_module.metrics[self.phase][0].clone()]
            self.post_init(trainer, pl_module)

    def on_phase_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx
    ):
        def reshape_tensor(tensor):
            return tensor.reshape(
                self.current_n + 1, -1, *output.shape[1:]
            )

        output = outputs['output']
        target = outputs['target']
        output = self.reduce(reshape_tensor(output)) if isinstance(output, torch.Tensor) else {
            key: self.reduce(reshape_tensor(output[key])) for key in output
        }
        target = reshape_tensor(target)[0] if isinstance(target, torch.Tensor) else {
            key: reshape_tensor(target[key])[0] for key in target
        }
        outputs['output'] = output
        outputs['target'] = target
        if self.phase != 'predict':
            output, target = self.metric_formatting(outputs=output, targets=target)
            if len(self.metrics) <= dataloader_idx:
                self.metrics.append(self.metrics[-1].clone())
            self.metrics[dataloader_idx].update(output, target)

    def on_phase_end(
            self,
            trainer,
            pl_module
    ):
        if self.phase != 'predict':
            for dataloader_idx in range(len(self.metrics)):
                prefix = f'{self.phase}_{dataloader_idx}' if dataloader_idx > 0 else self.phase
                metrics = self.metrics[dataloader_idx].compute()
                self.metrics[dataloader_idx].reset()
                for metric_name in metrics:
                    pl_module.formated_log(
                        f'{prefix}_tta[n={self.n} method={self.augmentation_method}]/{metric_name}',
                        metrics[metric_name],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True
                    )

    def metric_formatting(
            self,
            outputs: Any,
            targets: Any
    ) -> Tuple:
        """
        Preparing before metric pass. On default, return passed values.

        Attributes
        ----------
        outputs: Any
            Output from model

        targets: Any
            Targets from batch

        Returns
        ----------
        Tuple
            Formatted outputs and targets
        """
        return outputs, targets

    def reduce(
            self,
            tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Abstract method for reducing of results.

        Attributes
        ----------
        tensor: torch.Tensor
            Any tensor with size [batch_size X ...]

        Returns
        ----------
        torch.Tensor
            Reduced tensor

        """
        raise NotImplementedError

    def augment(
            self,
            sample: Dict,
            augmentation
    ) -> Dict:
        """
        Abstract method for augmentation apply.

        Attributes
        ----------
        sample: Dict
            Any sample of batch

        augmentation
            Transform object

        Returns
        ----------
        Dict
            Augmented sample
        """
        raise NotImplementedError

    def preprocessing(
            self,
            sample: Dict,
            dataloader_idx: int = 0
    ) -> Dict:
        """
        Abstract method for preprocessing sample

        Attributes
        ----------
        sample: Dict
            Any sample of batch

        dataloader_idx: int
            Index of dataloader

        Returns
        ----------
        Dict
            Preprocessed sample
        """
        return sample

    def postprocessing(
            self,
            sample: Dict,
            dataloader_idx: int = 0
    ) -> Dict:
        """
        Abstract method for postprocessing sample

        Attributes
        ----------
        sample: Dict
            Any sample of batch

        dataloader_idx: int
            Index of dataloader

        Returns
        ----------
        Dict
            Postprocessed sample
        """
        return sample

    def __augmentation_generator(self):
        if self.augmentation_method == 'first':
            return (augmentation for augmentation in self.augmentations[:self.n])
        elif self.augmentation_method == 'random':
            augmentations = np.random.choice(self.augmentations, self.n)
            return (augmentation for augmentation in augmentations)
        else:
            return (augmentation for augmentation in self.augmentations)

    def __collate_fn(self, dataloader_idx):
        def collate_fn_wrapper(batch):
            # TODO collate_fn_wrapper multiprocessing optimization
            batch_size = len(batch)
            samples = [
                self.preprocessing(_, dataloader_idx) for _ in batch
            ]
            augmented_samples = []
            augmentations = self.__augmentation_generator()
            for augmentation in augmentations:
                for sample in samples:
                    augmented_samples.append(self.augment(sample, augmentation))
            samples = samples + augmented_samples
            samples = [self.postprocessing(sample, dataloader_idx) for sample in samples]

            batch = self.collate_fns[dataloader_idx](samples)
            batch['batch_size'] = batch_size
            return batch

        return collate_fn_wrapper

    def on_validation_start(
            self, trainer, pl_module
    ):
        if self.phase == 'val':
            self.on_phase_start(trainer, pl_module)

    def on_validation_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx
    ):
        if self.phase == 'val':
            self.on_phase_batch_end(
                trainer,
                pl_module,
                outputs,
                batch,
                batch_idx,
                dataloader_idx
            )

    def on_test_start(
            self, trainer, pl_module
    ):
        if self.phase == 'test':
            self.on_phase_start(trainer, pl_module)

    def on_test_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx
    ):
        if self.phase == 'test':
            self.on_phase_batch_end(
                trainer,
                pl_module,
                outputs,
                batch,
                batch_idx,
                dataloader_idx
            )

    def on_predict_start(
            self, trainer, pl_module
    ):
        if self.phase == 'predict':
            self.on_phase_start(trainer, pl_module)

    def on_predict_batch_end(
            self,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx
    ):
        if self.phase == 'predict':
            self.on_phase_batch_end(
                trainer,
                pl_module,
                outputs,
                batch,
                batch_idx,
                dataloader_idx
            )

    def on_validation_epoch_end(
            self,
            trainer,
            pl_module
    ):
        if self.phase == 'val':
            self.on_phase_end(trainer, pl_module)

    def on_test_epoch_end(
            self,
            trainer,
            pl_module
    ):
        if self.phase == 'test':
            self.on_phase_end(trainer, pl_module)

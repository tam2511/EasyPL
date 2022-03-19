import torch
from pytorch_lightning.callbacks import Callback
from typing import List, Dict


class BaseTestTimeAugmentation(Callback):
    """ Base callback for test-time-augmentation """

    def __init__(
            self,
            n: int,
            augmentations: List,
            augmentation_method: str = 'first',
            phase='val'
    ):
        """
        :param augmentations: list of augmentation transforms
        """
        super().__init__()
        self.n = n
        self.augmentations = augmentations
        self.augmentation_method = augmentation_method
        self.phase = phase

        self.data_keys = None
        self.collate_fns = []
        self.metrics = []

    def post_init(self, trainer, pl_module):
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
                self.n + 1, -1, *output.shape[1:]
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
                        f'{prefix}/{metric_name}_tta[n={self.n} method={self.augmentation_method}]',
                        metrics[metric_name],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True
                    )

    def metric_formatting(self, outputs, targets):
        return outputs, targets

    def reduce(self, tensor):
        raise NotImplementedError

    def augment(self, sample: Dict, augmentation) -> Dict:
        raise NotImplementedError

    def preprocessing(self, sample: Dict, dataloader_idx: int) -> Dict:
        return sample

    def postprocessing(self, sample: Dict, dataloader_idx: int) -> Dict:
        return sample

    def __collate_fn(self, dataloader_idx):
        def collate_fn_wrapper(batch):
            # TODO collate_fn_wrapper multiprocessing optimization
            batch_size = len(batch)
            samples = [
                self.preprocessing(_, dataloader_idx) for _ in batch
            ]
            augmented_samples = []
            for augmentation in self.augmentations:
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

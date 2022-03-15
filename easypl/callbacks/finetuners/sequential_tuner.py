from pytorch_lightning.callbacks import BaseFinetuning


class SequentialFinetune(BaseFinetuning):
    """Callback for sequence unfreezing model"""

    def __init__(
            self,
            sequence: dict
    ):
        """
        :param sequence: dict of dicts. example: {0: {'layers': ['block1.layer_name1', ...]}, ...,
         12: {'layers: ['block12.layer_name13', ...]}, 14: {'layers: ['block14.layer_name3', ...], 'lr_gamma': 0.1}}
        """
        super().__init__()
        self.sequence = sequence

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.model, train_bn=False)
        if '0' not in self.sequence:
            raise ValueError('SequentialFinetune can work only with sequence which containing \'0\'')
        self.make_trainable([pl_module.model.get_submodule(layer_name) for layer_name in self.sequence['0']['layers']])

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        if str(current_epoch) not in self.sequence or current_epoch == 0:
            return
        subseq = self.sequence[str(current_epoch)]['layers']
        modules = [pl_module.model.get_submodule(layer_name) for layer_name in subseq]
        self.make_trainable(modules)
        trainable_params = self.filter_params(modules, train_bn=True, requires_grad=True)
        trainable_params = self.filter_on_optimizer(optimizer, trainable_params)
        last_group = optimizer.param_groups[-1]
        params = {param: last_group[param] for param in last_group if param != 'params'}
        for param in params:
            if (param.find('lr') > -1) and 'lr_gamma' in subseq:
                params[param] *= subseq['lr_gamma']
        params['params'] = trainable_params
        optimizer.add_param_group(params)

from torch.utils.data import Dataset


class DebugDataset(Dataset):
    '''
    Abstract wrapper for pytorch Dataset. You can override "raw" method and use him in "__getitem__" method for
    code economy.
    '''

    def raw(
            self,
            index: int
    ) -> dict:
        '''
        Getting data without transforms
        :param index: index of dataset
        :return: data without transforms in dict format
        '''
        raise NotImplementedError

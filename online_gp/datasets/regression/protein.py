import torch
import os
from scipy.io import loadmat
from torch.utils.data import TensorDataset, random_split


class Protein(object):
    def __init__(self, dataset_dir=None, subsample_ratio=1.0, test_ratio=0.1, split_seed=0, **kwargs):
        if dataset_dir is None:
            self.dataset_dir = '/datasets/uci/protein'
        else:
            self.dataset_dir = dataset_dir
        self.train_dataset, self.test_dataset = self._preprocess(subsample_ratio, test_ratio, split_seed)

    def _preprocess(self, subsample_ratio, test_ratio, split_seed):
        file_path = os.path.join(self.dataset_dir, 'protein.mat')
        data = loadmat(file_path)['data']
        inputs = torch.tensor(data[:, :-1], dtype=torch.get_default_dtype())
        targets = torch.tensor(data[:, -1], dtype=torch.get_default_dtype()).unsqueeze(-1)

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        input_max, _ = inputs.max(0)
        input_min, _ = inputs.min(0)
        input_range = input_max - input_min
        inputs = 2 * ((inputs - input_min) / input_range - 0.5)
        targets = (targets - targets.mean(0)) / targets.std(0)

        dataset = TensorDataset(inputs, targets)
        generator = torch.Generator().manual_seed(split_seed)
        num_samples = int(subsample_ratio * len(dataset))
        dataset, _ = random_split(dataset, [num_samples, len(dataset) - num_samples],
                                  generator=generator)
        dataset = TensorDataset(*dataset[:])

        num_test = int(test_ratio * len(dataset))
        train_dataset, test_dataset = random_split(dataset, [len(dataset) - num_test, num_test],
                                                   generator=generator)
        return train_dataset, test_dataset
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split


class Banana(object):
    def __init__(self, dataset_dir=None, subsample_ratio=1.0, split_seed=0, **kwargs):
        if dataset_dir is None:
            self.dataset_dir = "https://raw.githubusercontent.com/thangbui/streaming_sparse_gp/master/data"
        else:
            self.dataset_dir = dataset_dir
        self.train_dataset, self.test_dataset = self._preprocess(subsample_ratio, split_seed)

    def _get_raw_data(self, train):
        input_path = 'banana_train_x.txt' if train else 'banana_test_x.txt'
        target_path = 'banana_train_y.txt' if train else 'banana_test_y.txt'
        inputs = pd.read_csv(os.path.join(self.dataset_dir, input_path), header=None)
        targets = pd.read_csv(os.path.join(self.dataset_dir, target_path), header=None)
        inputs = torch.tensor(inputs.values).float().view(-1, 2)
        targets = torch.tensor(targets.values).long().view(-1)
        targets[targets < 0] = 0
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        return inputs, targets

    def _preprocess(self, subsample_ratio, split_seed):
        train_inputs, train_targets = self._get_raw_data(train=True)
        test_inputs, test_targets = self._get_raw_data(train=False)

        all_inputs = torch.cat([train_inputs, test_inputs])
        input_min, _ = all_inputs.min(0)
        input_max, _ = all_inputs.max(0)
        input_range = input_max - input_min

        train_inputs = 2 * ((train_inputs - input_min) / input_range - 0.5)
        test_inputs = 2 * ((test_inputs - input_min) / input_range - 0.5)

        train_dataset = TensorDataset(train_inputs, train_targets)
        generator = torch.Generator().manual_seed(split_seed)
        num_samples = int(subsample_ratio * len(train_dataset))
        train_dataset, _ = random_split(train_dataset, [num_samples, len(train_dataset) - num_samples],
                                        generator=generator)

        test_dataset = TensorDataset(test_inputs, test_targets)
        num_samples = int(subsample_ratio * len(test_dataset))
        test_dataset, _ = random_split(test_dataset, [num_samples, len(test_dataset) - num_samples],
                                       generator=generator)
        return train_dataset, test_dataset

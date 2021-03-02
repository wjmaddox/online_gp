import os
from sklearn.datasets import load_svmlight_file
import torch
from torch.utils.data import TensorDataset, random_split


class SVMGuide1(object):
    def __init__(self, dataset_dir, subsample_ratio=1.0, split_seed=0, **kwargs):
        self.dataset_dir = dataset_dir
        self.train_dataset, self.test_dataset = self._preprocess(subsample_ratio, split_seed)

    def _preprocess(self, subsample_ratio, split_seed):
        train_path = os.path.join(self.dataset_dir, 'train.libsvm')
        test_path = os.path.join(self.dataset_dir, 'train.libsvm')

        train_inputs, train_targets = load_svmlight_file(train_path)
        test_inputs, test_targets = load_svmlight_file(test_path)

        train_inputs, test_inputs = train_inputs.todense(), test_inputs.todense()
        train_inputs, train_targets = torch.tensor(train_inputs).float(), torch.tensor(train_targets).long()
        test_inputs, test_targets = torch.tensor(test_inputs).float(), torch.tensor(test_targets).long()

        if torch.cuda.is_available():
            train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
            test_inputs, test_targets = test_inputs.cuda(), test_targets.cuda()

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


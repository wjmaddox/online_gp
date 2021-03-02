import os
import torch
from torch.utils.data import TensorDataset
from online_gp.utils.cuda import try_cuda
from online_gp.utils.random import shuffle_tensors


class HopperV2(object):
    def __init__(self, dataset_dir=None, subsample_ratio=1.0, split_seed=0, shuffle=False, **kwargs):
        if dataset_dir is None:
            self.dataset_dir = '/datasets/mujoco/Hopper-v2'
        else:
            self.dataset_dir = dataset_dir

        self.train_dataset, self.test_dataset = self._preprocess(subsample_ratio, split_seed, shuffle)

    def _preprocess(self, subsample_ratio, split_seed, shuffle):
        train_x = torch.load(os.path.join(self.dataset_dir, 'train_x.pkl'))
        train_y = torch.load(os.path.join(self.dataset_dir, 'train_y.pkl'))
        test_x = torch.load(os.path.join(self.dataset_dir, 'test_x.pkl'))
        test_y = torch.load(os.path.join(self.dataset_dir, 'test_y.pkl'))

        train_x, train_y = try_cuda(train_x, train_y)
        test_x, test_y = try_cuda(test_x, test_y)

        if shuffle:
            train_x, train_y = shuffle_tensors(train_x, train_y)
            test_x, test_y = shuffle_tensors(test_x, test_y)

        num_train = int(subsample_ratio * train_x.size(0))
        num_test = int(subsample_ratio * test_x.size(0))

        train_x, train_y = train_x[:num_train], train_y[:num_train]
        test_x, test_y = test_x[:num_test], test_y[:num_test]

        return TensorDataset(train_x, train_y), TensorDataset(test_x, test_y)


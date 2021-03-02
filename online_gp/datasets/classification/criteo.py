import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split
from online_gp.utils.data import balance_classes


class Criteo(object):
    def __init__(self, dataset_dir, num_rows, **kwargs):
        if dataset_dir is None:
            self.dataset_dir = "/datasets/criteo"
        else:
            self.dataset_dir = dataset_dir
        self.train_dataset, self.test_dataset = self._preprocess(num_rows)

    def _preprocess(self, num_rows):
        file_path = os.path.join(self.dataset_dir, 'train.txt')
        criteo_df = pd.read_csv(file_path, sep='\t', header=None, low_memory=True, memory_map=True,
                                nrows=num_rows)

        labels = criteo_df[0]
        int_features = criteo_df[list(range(1, 14))]
        cat_features = criteo_df[list(range(14, 40))]

        # log transform large values, standardize, and mean-fill
        int_features = int_features.applymap(lambda x: np.log(x) ** 2 if x > 2 else x)
        int_features = (int_features - int_features.mean()) / int_features.std()
        int_features.fillna(0, inplace=True)

        # TODO drop any categories in the test set that do not appear in the train set
        # drop low-frequency categories, convert to one-hot
        cat_features = cat_features.apply(lambda x: x.mask(x.map(x.value_counts()) < 4, float('NaN')))
        cat_features = cat_features.apply(lambda x: x.astype('category'))
        cat_features = pd.get_dummies(cat_features, dummy_na=True)

        all_features = np.concatenate([int_features.values, cat_features.values], axis=1)
        all_features = torch.tensor(all_features).float()
        labels = torch.tensor(labels.values).long()
        row_perm = torch.randperm(all_features.size(0))
        all_features = all_features[row_perm]
        labels = labels[row_perm]

        if torch.cuda.is_available():
            all_features, labels = all_features.cuda(), labels.cuda()

        num_train = int(all_features.size(0) * 0.9)
        num_test = all_features.size(0) - num_train
        dataset = TensorDataset(all_features, labels)
        train_dataset, test_dataset = random_split(dataset, [num_train, num_test])
        train_dataset = balance_classes(train_dataset, num_classes=2)
        test_dataset = balance_classes(test_dataset, num_classes=2)

        return train_dataset, test_dataset

import numpy as np
import pandas as pd
import torch


def get_datasets(name):
    if name == 'banana':
        train_x = pd.read_csv(
            "https://raw.githubusercontent.com/thangbui/streaming_sparse_gp/master/data/banana_train_x.txt",
            header=None)
        train_y = pd.read_csv(
            "https://raw.githubusercontent.com/thangbui/streaming_sparse_gp/master/data/banana_train_y.txt",
            header=None)
        train_x = torch.tensor(train_x.values).float().view(-1, 2)
        train_y = torch.tensor(train_y.values).long().view(-1)
        train_y[train_y < 0] = 0
        if torch.cuda.is_available():
            train_x, train_y = train_x.cuda(), train_y.cuda()
        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)

        test_x = pd.read_csv(
            "https://raw.githubusercontent.com/thangbui/streaming_sparse_gp/master/data/banana_test_x.txt",
            header=None)
        test_y = pd.read_csv(
            "https://raw.githubusercontent.com/thangbui/streaming_sparse_gp/master/data/banana_test_y.txt",
            header=None)
        test_x = torch.tensor(test_x.values).float().view(-1, 2)
        test_y = torch.tensor(test_y.values).long().view(-1)
        test_y[test_y < 0] = 0
        if torch.cuda.is_available():
            test_x, test_y = test_x.cuda(), test_y.cuda()
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

    if name == 'criteo':
        criteo_df = pd.read_csv('../data/criteo/train.txt', sep='\t', header=None, low_memory=True, memory_map=True,
                                nrows=1000)

        labels = criteo_df[0]
        int_features = criteo_df[list(range(1, 14))]
        cat_features = criteo_df[list(range(14, 40))]

        # log transform large values, standardize, and mean-fill
        int_features = int_features.applymap(lambda x: np.log(x) ** 2 if x > 2 else x)
        int_features = (int_features - int_features.mean()) / int_features.std()
        int_features.fillna(0, inplace=True)

        # TODO drop any categories in the test set that do not appear in the train set
        # drop low-frequency categories, convert to one-hot
        # cat_features = cat_features.apply(lambda x: x.mask(x.map(x.value_counts()) < 8, float('NaN')))
        cat_features = cat_features.apply(lambda x: x.astype('category'))
        cat_features = pd.get_dummies(cat_features, dummy_na=True)

        all_features = np.concatenate([int_features.values, cat_features.values], axis=1)
        all_features = torch.tensor(all_features).float()
        labels = torch.tensor(labels.values).long()
        row_perm = torch.randperm(all_features.size(0))
        all_features = all_features[row_perm]
        labels = labels[row_perm]

        num_train = int(all_features.size(0) * 0.9)
        num_test = all_features.size(0) - num_train
        dataset = torch.utils.data.TensorDataset(
            all_features,
            labels
        )
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])
        train_dataset = balance_classes(train_dataset, num_classes=2)
        test_dataset = balance_classes(test_dataset, num_classes=2)

    return train_dataset, test_dataset


def balance_classes(dataset, num_classes=2):
    inputs, targets = dataset[:]
    num_train = inputs.size(0)
    balanced_inputs, balanced_targets = [], []
    for class_idx in range(num_classes):
        num_class_examples = num_train // num_classes
        mask = (targets == class_idx)
        masked_inputs, masked_targets = inputs[mask], targets[mask]
        idxs = torch.randint(masked_inputs.size(0), (num_class_examples,))
        balanced_inputs.append(masked_inputs[idxs])
        balanced_targets.append(masked_targets[idxs])
    balanced_inputs = torch.cat(balanced_inputs)
    balanced_targets = torch.cat(balanced_targets)
    row_perm = torch.randperm(balanced_inputs.size(0))
    balanced_dataset = torch.utils.data.TensorDataset(
        balanced_inputs[row_perm],
        balanced_targets[row_perm]
    )
    return balanced_dataset

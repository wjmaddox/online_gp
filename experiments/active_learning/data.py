import torch

import pandas as pd
import numpy as np


def subsample(train_x, train_y, train_y_var, n_samp):
    """Selects n_samp random rows from training data"""
    idx = np.random.permutation(range(len(train_x)))[:n_samp]
    return train_x[idx], train_y[idx], train_y_var[idx]


def unitize(x):
    """Puts design space on a unit cube"""
    x1 = x - x.min()
    return x1 / x1.max()


def prepare_data(data_loc, num_init, num_total=None, test_is_year=True, seed=0):
    df = pd.read_hdf(data_loc, "full")
    df = df[df["is_ng"] == 1]
    if test_is_year:
        is_test = torch.from_numpy((df["year"] == 2017).values)
    else:
        df = df[df["year"] == 2017]
        df.reset_index(inplace=True)
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(df.shape[0])
        test_set = shuffled_indices[-10000:]
        is_test = np.array([x in test_set for x in np.arange(40517)])
        
    
    if test_is_year:
        tokeep = ["longitude", "latitude", "year"]
    else:
        tokeep = ["longitude", "latitude"]

    all_x = torch.from_numpy(
        df[tokeep].values
    ).float()
    lon_lims = (all_x[:, 0].min().item(), all_x[:, 0].max().item())
    lat_lims = (all_x[:, 1].min().item(), all_x[:, 1].max().item())
    extent = lon_lims + lat_lims

    all_x[:, 0] = unitize(all_x[:, 0])
    all_x[:, 1] = unitize(all_x[:, 1])
    if test_is_year:
        all_x[:, 2] = unitize(all_x[:, 2])

    all_y = torch.from_numpy(df["mean"].values).float()
    all_y_var = torch.from_numpy(df["std_dev"].values).pow(2).float()

    train_x, train_y, train_y_var = (
        all_x[~is_test],
        all_y[~is_test],
        all_y_var[~is_test],
    )
    test_x, test_y, test_y_var = (
        all_x[is_test],
        all_y[is_test],
        all_y_var[is_test],
    )

    # filter out the year for the regression experiment
    if test_is_year:
        train_y, train_y_var = (
            train_y[train_x[:, 2] == 0.0],
            train_y_var[train_x[:, 2] == 0.0],
        )
        train_x = train_x[train_x[:, 2] == 0.0, :2]
    train_y_var += 1e-6  # so there's no zeros

    init_x, init_y, init_y_var = (
        train_x[:num_init],
        train_y[:num_init],
        train_y_var[:num_init],
    )
    # init_y_var += 1e-6

    train_x, train_y, train_y_var = (
        train_x[:num_total],
        train_y[:num_total],
        train_y_var[:num_total],
    )
    return (
        {"x": init_x, "y": init_y, "y_var": init_y_var},
        {"x": train_x, "y": train_y, "y_var": train_y_var},
        {"x": test_x, "y": test_y, "y_var": test_y_var},
    )
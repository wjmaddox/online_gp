import torch
import numpy as np
from unittest import TestCase, main
import gpytorch
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ZeroMean
from gpytorch.models import ExactGP
from copy import deepcopy

from online_gp.models.batched_fixed_noise_online_gp import FixedNoiseOnlineSKIGP

# this unit test is only for testing the class based updates in FixedNoiseOnlineSKIGP
class TestFixedNoiseOnlineSKIGP(TestCase):
    def setUp_nonBatch(self):
        train_x = torch.rand(10, 1)
        train_y = torch.sin(3.0 * train_x)
        train_y_var = 0.01 * train_y ** 2

        model = FixedNoiseOnlineSKIGP(
            train_inputs=train_x[:5],
            train_targets=train_y[:5],
            train_noise_term=train_y_var[:5].t(),
            grid_bounds=torch.tensor([[0.0, 1.0]]),
            grid_size=10,
        )

        return model, train_x[5:], train_y[:, 5:], train_y_var[:, 5:]

    def test_forwards_and_initial_cache(self):
        model, next_x, _, _ = self.setUp_nonBatch()

        model.train()
        dist = model(*model.train_inputs)

        # initial non-batch shape forwards pass
        self.assertLessEqual((dist.mean - torch.zeros(5)).norm(), 1e-5)
        self.assertEqual(dist.mean.shape, torch.Size((5,)))
        self.assertEqual(dist.covariance_matrix.shape, torch.Size((5, 5)))

        # now check posterior functionality
        model.eval()
        dist = model(next_x)
        # initial batched forwards pass shapes
        self.assertEqual(dist.mean.shape, torch.Size((5,)))
        self.assertEqual(dist.covariance_matrix.shape, torch.Size((5, 5)))

    def setUp_Batch(self):
        train_x = torch.rand(10, 1)
        train_y = torch.stack((torch.sin(3.0 * train_x), torch.sin(5.0 * train_x)))[
            ..., 0
        ].t()
        train_y_var = 0.01 * train_y ** 2
        model = FixedNoiseOnlineSKIGP(
            train_inputs=train_x[:5],
            train_targets=train_y[:5],
            train_noise_term=train_y_var[:5],
            grid_bounds=torch.tensor([[0.0, 1.0]]),
            grid_size=10,
        )

        return model, train_x[5:], train_y[5:], train_y_var[5:]

    def _test_forwards_shaping(self, model, X, is_train):
        if type(X) == list:
            X = X[0]
        dist = model(X)

        model_batch_shape = torch.Size((model._batch_shape,)) if type(model._batch_shape) is not torch.Size else model._batch_shape
        if is_train:
            model_num_data = model.num_data
        else:
            model_num_data = X.shape[-2]

        if len(model_batch_shape) > 0:
            dims = [model_batch_shape[0], model_num_data]
        else:
            dims = [model_num_data]

        if is_train:
            self.assertLessEqual((dist.mean - torch.zeros(*dims)).norm(), 1e-5)

        # initial batched forwards pass shapes
        self.assertEqual(dist.mean.shape, torch.Size(dims))
        self.assertEqual(dist.covariance_matrix.shape, torch.Size(dims + [dims[-1]]))

    def test_batch_forwards_and_initial_cache(self):
        model, next_x, _, _ = self.setUp_Batch()

        model.train()
        self._test_forwards_shaping(model, model.train_inputs, is_train=True)

        # check that the cache shapes are okay
        self.assertEqual(model.current_qmatrix.shape, torch.Size((2, 10, 10)))
        self.assertEqual(model._kernel_cache["WtW"].shape, torch.Size((2, 10, 10)))
        self.assertEqual(model._kernel_cache["D_logdet"].shape, torch.Size((2,)))
        self.assertEqual(
            model._kernel_cache["response_cache"].shape, torch.Size((2, 1, 1))
        )
        self.assertEqual(
            model._kernel_cache["interpolation_cache"].shape, torch.Size((2, 10, 1))
        )

        # now check posterior functionality
        model.eval()
        self._test_forwards_shaping(model, next_x, is_train=False)

    def test_batch_cache_updating(self):
        model, test_x, test_y, test_y_var = self.setUp_Batch()
        orig_train_inputs = model.num_data

        new_model = model.condition_on_observations(
            X=test_x[0].unsqueeze(0),
            Y=test_y[0].unsqueeze(0),
            noise=test_y_var[0].unsqueeze(0),
            inplace=False,
        )

        new_model.train()
        self.assertEqual(new_model.num_data, orig_train_inputs + 1)
        self._test_forwards_shaping(new_model, new_model.train_inputs, is_train=True)

        model.condition_on_observations(
            X=test_x[0].unsqueeze(0),
            Y=test_y[0].unsqueeze(0),
            noise=test_y_var[0].unsqueeze(0),
            inplace=True,
        )
        model.eval()
        self.assertEqual(model.num_data, orig_train_inputs + 1)
        self._test_forwards_shaping(model, test_x, is_train=False)


if __name__ == "__main__":
    main()

import torch
import gpytorch
from gpytorch.kernels import GridInterpolationKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from unittest import TestCase, main
import numpy as np
from gpytorch.models.exact_prediction_strategies import InterpolatedPredictionStrategy
from online_gp.models.sherman_morrison_prediction_strategy import (
    ShermanMorrisonOnlineStrategy,
)
from online_gp.models.interpolated_prediction_strategy_with_fantasy import (
    InterpolatedPredictionStrategyWithFantasy,
)


class OnlinePredictionStrategyTest(TestCase):
    def setUp(self):
        self.xs = torch.tensor([0.20, 0.30, 0.40, 0.10, 0.70], dtype=torch.double)
        self.kernel = GridInterpolationKernel(
            RBFKernel(), grid_size=4, grid_bounds=[(-0.4, 1.4)]
        ).double()
        self.mean_vec = torch.sin(self.xs).double() * 0.0
        self.labels = torch.sin(self.xs) + torch.tensor(
            [0.1, 0.2, -0.1, -0.2, -0.2], dtype=torch.double
        )
        self.lik = GaussianLikelihood().double()
        self.lik.noise = 0.1
        self.train_train_covar = self.kernel(self.xs).evaluate_kernel()
        self.distr = MultivariateNormal(self.mean_vec, self.train_train_covar)

        e_distr = MultivariateNormal(self.mean_vec, self.train_train_covar)
        e_lik = GaussianLikelihood().double()
        e_lik.noise = 0.1
        self.expected_strategy = InterpolatedPredictionStrategyWithFantasy(
            self.xs, e_distr, self.labels, e_lik
        )
        self.strategy = ShermanMorrisonOnlineStrategy(
            self.xs, self.distr, self.labels, self.lik
        )

        self.new_points = torch.tensor([0.5, 0.8], dtype=torch.double)
        self.test_mean = torch.sin(self.new_points) * 0.0
        self.test_train_covar = self.kernel(self.new_points, self.xs).evaluate_kernel()
        self.test_test_covar = self.kernel(
            self.new_points, self.new_points
        ).evaluate_kernel()

    def tearDown(self):
        import os
        import glob

        for f in glob.glob("./*.csv"):
            os.remove(f)

    def test_predictive_mean(self):
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples(), gpytorch.settings.max_cholesky_size(
            0
        ), gpytorch.settings.cg_tolerance(
            1e-2
        ):
            actual_mean = self.strategy.exact_predictive_mean(
                self.test_mean, self.test_train_covar
            )

            expected_mean = self.expected_strategy.exact_predictive_mean(
                self.test_mean, self.test_train_covar
            )

            np.testing.assert_allclose(
                actual_mean.detach().numpy(), expected_mean.detach().numpy(), rtol=1e-5
            )

    def test_predictive_covar(self):
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples(), gpytorch.settings.fast_computations(), gpytorch.settings.max_cholesky_size(
            0
        ), gpytorch.settings.cg_tolerance(
            1e-2
        ):

            actual_covar = self.strategy.exact_predictive_covar(
                self.test_test_covar, self.test_train_covar
            )

            expected_covar = self.expected_strategy.exact_predictive_covar(
                self.test_test_covar, self.test_train_covar
            )

            np.testing.assert_allclose(
                actual_covar.detach().numpy(), expected_covar.detach().numpy(), rtol=1e-5
            )

    def test_woodbury_cache(self):
        with gpytorch.settings.use_toeplitz():
            woodbury_cache = self.strategy.woodbury_cache

        covar = self.train_train_covar.base_lazy_tensor.evaluate().double()
        inv_covar = torch.inverse(covar)

        W_T = self.train_train_covar._sparse_left_interp_t(
            self.train_train_covar.left_interp_indices,
            self.train_train_covar.left_interp_values,
        ).to_dense()
        w_t_w = W_T.matmul(W_T.t())

        expected_woodbury = self.lik.noise * inv_covar + w_t_w
        np.testing.assert_allclose(
            woodbury_cache.detach().numpy(), expected_woodbury.detach().numpy(), rtol=1e-3
        )

    def test_fantasy_strategy(self):
        fant_x = torch.tensor([0.45], dtype=torch.double)
        fant_y = torch.sin(fant_x) + torch.tensor(0.12, dtype=torch.double)

        full_inputs = torch.cat([self.xs, fant_x], dim=-1)
        full_targets = torch.cat([self.labels, fant_y], dim=-1)
        full_output = MultivariateNormal(
            torch.sin(full_inputs) * 0.0, self.kernel(full_inputs)
        )

        with gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples(), gpytorch.settings.max_cholesky_size(
            0
        ), gpytorch.settings.cg_tolerance(
            1e-2
        ), gpytorch.settings.min_preconditioning_size(
            0
        ), gpytorch.settings.max_preconditioner_size(
            128
        ), gpytorch.settings.max_root_decomposition_size(
            128
        ):

            fant_strat = self.strategy.get_fantasy_strategy(
                fant_x, fant_y, full_inputs, full_targets, full_output
            )
            fant_strat_offline_strat = self.expected_strategy.get_fantasy_strategy(
                fant_x, fant_y, full_inputs, full_targets, full_output
            )

            full_output = MultivariateNormal(
                torch.sin(full_inputs) * 0.0, self.kernel(full_inputs).evaluate_kernel()
            )
            full_strat = ShermanMorrisonOnlineStrategy(
                full_inputs, full_output, full_targets, likelihood=self.lik
            )

            actual_mean = fant_strat.exact_predictive_mean(
                self.test_mean, self.test_train_covar
            )
            actual_mean_offline = fant_strat_offline_strat.exact_predictive_mean(
                self.test_mean, self.test_train_covar
            )

            expected_mean = full_strat.exact_predictive_mean(
                self.test_mean, self.test_train_covar
            )

            np.testing.assert_allclose(
                actual_mean.detach().numpy(), expected_mean.detach().numpy(), rtol=1e-5
            )
            np.testing.assert_allclose(
                actual_mean_offline.detach().numpy(),
                expected_mean.detach().numpy(),
                rtol=1e-5,
            )

            actual_covar = fant_strat.exact_predictive_covar(
                self.test_test_covar, self.test_train_covar
            )
            actual_covar_offline = fant_strat_offline_strat.exact_predictive_covar(
                self.test_test_covar, self.test_train_covar
            )

            expected_covar = full_strat.exact_predictive_covar(
                self.test_test_covar, self.test_train_covar
            )

            np.testing.assert_allclose(
                actual_covar.detach().numpy(),
                expected_covar.detach().numpy(),
                rtol=1e-5,
                # atol=1e-6
            )

            np.testing.assert_allclose(
                actual_covar_offline.detach().numpy(),
                expected_covar.detach().numpy(),
                rtol=1e-5,
                # atol=1e-6
            )


if __name__ == "__main__":
    main()

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

from online_gp.kernels import GridInterpolationKernelWithFantasy
from online_gp.models import OnlineWoodburyGP

# from online_gp.likelihoods import WoodburyGaussianLikelihood
from online_gp.mlls import WoodburyExactMarginalLogLikelihood
from online_gp.settings import woodbury_sherman_morrison_inverse

torch.set_default_dtype(torch.float64)
woodbury_sherman_morrison_inverse.state = (
    True  # only test the sherman morrison stuff with direct woodbury comps.
)
# gpytorch.settings.deterministic_probes._set_state(True)
gpytorch.settings.max_lanczos_quadrature_iterations._global_value = 100
gpytorch.settings.num_trace_samples._set_value(1000)  # use a lot of samples so that
# it's unlikely we have errors due to
# randomness
gpytorch.settings.max_root_decomposition_size._set_value(100)
gpytorch.settings.cg_tolerance._set_value(0.01)
gpytorch.settings.eval_cg_tolerance._set_value(0.01)

# TODO: test with more data points.
class RegularExactGP(ExactGP):
    def __init__(
        self, train_inputs, train_targets, likelihood, covar_module, mean_module
    ):
        super(RegularExactGP, self).__init__(train_inputs, train_targets, likelihood)
        self.covar_module = covar_module
        self.mean_module = mean_module

    def forward(self, x1):
        mean = self.mean_module(x1)
        covar = self.covar_module(x1)
        return MultivariateNormal(mean, covar)

    def get_fantasy_model(self, inputs, targets, **kwargs):
        # Does this work?
        cached_kernel_mat = (
            self.covar_module._cached_kernel_mat
            if hasattr(self.covar_module, "_cached_kernel_mat")
            else None
        )
        del self.covar_module._cached_kernel_mat
        model = super().get_fantasy_model(inputs, targets, **kwargs)
        if cached_kernel_mat is not None:
            model.covar_module._cached_kernel_mat = cached_kernel_mat
        return model


class BaseTestCaseMixin(object):
    def _setUp(self):
        self.xs = torch.tensor([2.0, 3.0, 4.0, 1.0, 7.0], dtype=torch.double)
        # self.xs = torch.rand(100).double() * 14 - 2
        self.grid_size = 20
        self.kernel = GridInterpolationKernelWithFantasy(
            RBFKernel(), grid_size=self.grid_size, grid_bounds=[(-4.0, 14.0)]
        ).double()
        # self.lengthscale = np.random.rand()*10 + 0.1
        # self.noise_var = np.random.rand()*10 + 0.1
        self.lengthscale = 10.0
        self.noise_var = 0.01
        self.lr = 0.1

        self.mean_module = ZeroMean()
        self.labels = torch.sin(self.xs) + torch.tensor(
            [0.1, 0.2, -0.1, -0.2, -0.2], dtype=torch.double
        )
        # self.labels = torch.sin(self.xs) + torch.randn_like(self.xs)*0.1
        self.test_points = torch.tensor([5.0, 8.0], dtype=torch.double)

        self.new_points = torch.tensor([2.4, 4.7], dtype=torch.double)
        self.new_targets = torch.sin(self.new_points) + torch.tensor(
            [0.1, -0.15], dtype=torch.double
        )

        self.points_sequence = [
            self.xs,
            self.new_points,
            torch.tensor([2.3]),
            torch.tensor([4.1]),
            torch.tensor([4.3]),
        ]

        self.targets_sequence = [
            self.labels,
            self.new_targets,
            torch.sin(torch.tensor([2.3])),
            torch.sin(torch.tensor([4.1])) + 1,
            torch.sin(torch.tensor([4.3])),
        ]

    def tearDown(self):
        import os
        import glob

        for f in glob.glob("./*.csv"):
            os.remove(f)

    def test_init(self):
        # mostly make sure it doesn't crash
        lik = GaussianLikelihood()
        lik.noise = 1.0
        model = OnlineWoodburyGP(self.xs, self.labels, lik, self.kernel, self.mean_module)

    def test_train_mll_backprop(self):
        """This test is intended to test training without online observations"""

        def update(lengthscale, noise_var, xs, ys):
            r_lik = GaussianLikelihood()
            r_kernel = GridInterpolationKernelWithFantasy(
                RBFKernel(), grid_size=self.grid_size, grid_bounds=[(-4.0, 14.0)]
            ).double()
            r_model = RegularExactGP(xs, ys, r_lik, r_kernel, ZeroMean())

            lik = GaussianLikelihood()
            kernel = GridInterpolationKernelWithFantasy(
                RBFKernel(), grid_size=self.grid_size, grid_bounds=[(-4.0, 14.0)]
            ).double()
            model = OnlineWoodburyGP(xs, ys, lik, kernel, ZeroMean())

            r_model.covar_module.base_kernel.lengthscale = lengthscale
            model.covar_module.base_kernel.lengthscale = lengthscale

            r_model.likelihood.noise = noise_var
            model.likelihood.noise = noise_var

            r_model.train()
            r_optim = torch.optim.SGD(r_model.parameters(), self.lr)

            model.train()
            optim = torch.optim.SGD(model.parameters(), self.lr)

            with gpytorch.settings.fast_computations(), gpytorch.settings.max_cholesky_size(
                1
            ), gpytorch.settings.skip_logdet_forward():
                r_mll = ExactMarginalLogLikelihood(r_model.likelihood, r_model)
                r_train_output = r_model(r_model.train_inputs[0])
                r_mll_val = r_mll(r_train_output, r_model.train_targets)

                mll = WoodburyExactMarginalLogLikelihood(model.likelihood, model)
                train_output = model(model.train_inputs[0])
                mll_val = mll(train_output, model.train_targets)

                loss = -mll_val
                loss.backward()
                r_loss = -r_mll_val
                r_loss.backward()

            print(
                "online ls grad",
                model.covar_module.base_kernel.raw_lengthscale.grad.item(),
            )
            print(
                "ski ls grad",
                r_model.covar_module.base_kernel.raw_lengthscale.grad.item(),
            )

            print("online ls", model.covar_module.base_kernel.lengthscale.item())
            print("ski ls", r_model.covar_module.base_kernel.lengthscale.item())

            print("online noise grad", model.likelihood.raw_noise.grad.item())
            print("ski noise grad", r_model.likelihood.raw_noise.grad.item())

            print("online noise", model.likelihood.noise.item())
            print("ski noise", r_model.likelihood.noise.item())

            # Make sure the gradients are the same
            np.testing.assert_allclose(
                model.covar_module.base_kernel.raw_lengthscale.grad.item(),
                r_model.covar_module.base_kernel.raw_lengthscale.grad.item(),
                rtol=0.01,
                atol=0.01,
            )
            np.testing.assert_allclose(
                model.likelihood.raw_noise.grad.item(),
                r_model.likelihood.raw_noise.grad.item(),
                rtol=0.01,
                atol=0.01,
            )

            r_optim.step()
            r_optim.zero_grad()

            optim.step()
            optim.zero_grad()
            model.get_updated_hyper_strategy()

            # Make sure the values are the same
            np.testing.assert_allclose(
                model.covar_module.base_kernel.lengthscale.item(),
                r_model.covar_module.base_kernel.lengthscale.item(),
                rtol=0.01,
                atol=0.01,
            )
            np.testing.assert_allclose(
                model.likelihood.noise.item(),
                r_model.likelihood.noise.item(),
                rtol=0.01,
                atol=0.01,
            )

            # Verify the gradients are the same = 0
            np.testing.assert_allclose(
                model.covar_module.base_kernel.raw_lengthscale.grad.item(),
                r_model.covar_module.base_kernel.raw_lengthscale.grad.item(),
                rtol=0.01,
                atol=0.01,
            )
            np.testing.assert_allclose(
                model.likelihood.raw_noise.grad.item(),
                r_model.likelihood.raw_noise.grad.item(),
            )
            return r_model, model

        update(
            lengthscale=self.lengthscale,
            noise_var=self.noise_var,
            xs=self.xs,
            ys=self.labels,
        )

        larger_xs = torch.cat(self.points_sequence[:2])
        larger_ys = torch.cat(self.targets_sequence[:2])
        update(
            lengthscale=self.lengthscale,
            noise_var=self.noise_var,
            xs=larger_xs,
            ys=larger_ys,
        )

        still_larger_xs = torch.cat(self.points_sequence[:3])
        still_larger_ys = torch.cat(self.targets_sequence[:3])
        update(
            lengthscale=self.lengthscale,
            noise_var=self.noise_var,
            xs=still_larger_xs,
            ys=still_larger_ys,
        )

        still_larger_xs = torch.cat(self.points_sequence[:4])
        still_larger_ys = torch.cat(self.targets_sequence[:4])
        update(
            lengthscale=self.lengthscale,
            noise_var=self.noise_var,
            xs=still_larger_xs,
            ys=still_larger_ys,
        )

    def test_eval_mode(self):
        lik = GaussianLikelihood()
        lik.noise = self.noise_var
        model = OnlineWoodburyGP(self.xs, self.labels, lik, self.kernel, self.mean_module)
        model.covar_module.base_kernel.lengthscale = self.lengthscale
        model.eval()
        pred_nn = model(self.test_points)
        pred = model.likelihood(pred_nn)

        r_lik = GaussianLikelihood()
        r_lik.noise = self.noise_var
        r_kernel = GridInterpolationKernelWithFantasy(
            RBFKernel(), grid_size=self.grid_size, grid_bounds=[(-4.0, 14.0)]
        ).double()
        r_model = RegularExactGP(
            self.xs, self.labels, r_lik, r_kernel, deepcopy(self.mean_module)
        )
        r_model.covar_module.base_kernel.lengthscale = self.lengthscale
        r_model.eval()
        r_pred_nn = r_model(self.test_points)
        r_pred = r_model.likelihood(r_pred_nn)

        np.testing.assert_allclose(
            r_pred.mean.detach().numpy(), pred.mean.detach().numpy()
        )

        np.testing.assert_allclose(
            r_pred.covariance_matrix.detach().numpy(),
            pred.covariance_matrix.detach().numpy(),
        )

    def test_online_train_mll_backprop(self):
        """This test is intended to test consecutive observe-train-observe-train patterns"""
        r_lik = GaussianLikelihood()
        r_kernel = GridInterpolationKernelWithFantasy(
            RBFKernel(), grid_size=self.grid_size, grid_bounds=[(-4.0, 14.0)]
        ).double()
        r_model = RegularExactGP(self.xs, self.labels, r_lik, r_kernel, ZeroMean())

        lik = GaussianLikelihood()
        kernel = GridInterpolationKernelWithFantasy(
            RBFKernel(), grid_size=self.grid_size, grid_bounds=[(-4.0, 14.0)]
        ).double()
        model = OnlineWoodburyGP(self.xs, self.labels, lik, kernel, ZeroMean())

        def observe_and_update(
            r_model, model, lengthscale, noise_var, xs, ys, set_online=False
        ):
            r_model.covar_module.base_kernel.lengthscale = lengthscale
            if set_online:
                model.covar_module.base_kernel.lengthscale = lengthscale

            r_model.likelihood.noise = noise_var
            if set_online:
                model.likelihood.noise = noise_var

            r_model.eval()
            r_model(self.new_points)
            r_model = r_model.get_fantasy_model(xs, ys)
            r_model.train()
            r_optim = torch.optim.SGD(r_model.parameters(), self.lr)

            model.eval()
            model(self.new_points)
            model = model.get_online_model(xs, ys)
            model.train()
            optim = torch.optim.SGD(model.parameters(), self.lr)

            with gpytorch.settings.fast_computations(), gpytorch.settings.max_cholesky_size(
                1
            ), gpytorch.settings.skip_logdet_forward():
                r_mll = ExactMarginalLogLikelihood(r_model.likelihood, r_model)
                r_train_output = r_model(r_model.train_inputs[0])
                r_mll_val = r_mll(r_train_output, r_model.train_targets)

                mll = WoodburyExactMarginalLogLikelihood(lik, model)
                train_output = model(model.train_inputs[0])
                mll_val = mll(train_output, model.train_targets)

                np.testing.assert_allclose(r_mll_val.item(), mll_val.item(), rtol=1e-4)

                loss = -mll_val
                loss.backward()
                r_loss = -r_mll_val
                r_loss.backward()

            print(
                "online ls grad",
                model.covar_module.base_kernel.raw_lengthscale.grad.item(),
            )
            print(
                "ski ls grad",
                r_model.covar_module.base_kernel.raw_lengthscale.grad.item(),
            )

            print("online ls", model.covar_module.base_kernel.lengthscale.item())
            print("ski ls", r_model.covar_module.base_kernel.lengthscale.item())

            print("online noise grad", model.likelihood.raw_noise.grad.item())
            print("ski noise grad", r_model.likelihood.raw_noise.grad.item())

            print("online noise", model.likelihood.noise.item())
            print("ski noise", r_model.likelihood.noise.item())

            # Make sure the gradients are the same
            np.testing.assert_allclose(
                model.covar_module.base_kernel.raw_lengthscale.grad.item(),
                r_model.covar_module.base_kernel.raw_lengthscale.grad.item(),
                rtol=0.01,
                atol=0.01,
            )
            np.testing.assert_allclose(
                model.likelihood.raw_noise.grad.item(),
                r_model.likelihood.raw_noise.grad.item(),
                rtol=0.01,
                atol=0.01,
            )

            r_optim.step()
            r_optim.zero_grad()

            optim.step()
            optim.zero_grad()
            model.get_updated_hyper_strategy()

            # Make sure the values are the same
            np.testing.assert_allclose(
                model.covar_module.base_kernel.lengthscale.item(),
                r_model.covar_module.base_kernel.lengthscale.item(),
                rtol=0.01,
                atol=0.01,
            )
            np.testing.assert_allclose(
                model.likelihood.noise.item(),
                r_model.likelihood.noise.item(),
                rtol=0.01,
                atol=0.01,
            )

            # Verify the gradients are the same = 0
            np.testing.assert_allclose(
                model.covar_module.base_kernel.raw_lengthscale.grad.item(),
                r_model.covar_module.base_kernel.raw_lengthscale.grad.item(),
                rtol=0.01,
                atol=0.01,
            )
            np.testing.assert_allclose(
                model.likelihood.raw_noise.grad.item(),
                r_model.likelihood.raw_noise.grad.item(),
            )
            return r_model, model

        # dot = make_dot(mll_val, dict(model.named_parameters()))
        # dot.render('test-mll_graph.gv', view=True)
        # r_dot = make_dot(r_mll_val, dict(r_model.named_parameters()))
        # r_dot.render('test-r_mll_graph.gv', view=True)
        # # self.assertAlmostEqual(mll_val.item(), r_mll_val.item(), places=4)

        r_model, model = observe_and_update(
            r_model,
            model,
            self.lengthscale,
            self.noise_var,
            self.points_sequence[1],
            self.targets_sequence[1],
            set_online=True,
        )
        ls = deepcopy(model.covar_module.base_kernel.lengthscale.item())
        nv = deepcopy(model.likelihood.noise.item())

        r_model, model = observe_and_update(
            r_model, model, ls, nv, self.points_sequence[2], self.targets_sequence[2]
        )
        ls = deepcopy(model.covar_module.base_kernel.lengthscale.item())
        nv = deepcopy(model.likelihood.noise.item())

        r_model, model = observe_and_update(
            r_model, model, ls, nv, self.points_sequence[3], self.targets_sequence[3]
        )
        ls = deepcopy(model.covar_module.base_kernel.lengthscale.item())
        nv = deepcopy(model.likelihood.noise.item())

        observe_and_update(
            r_model, model, ls, nv, self.points_sequence[4], self.targets_sequence[4]
        )

    def test_mixed_online_train_mll_backprop(self):
        """This test is intended to sort out whether gradients are bad after one/more online observations
        Also, to test whether multiple grad updates affect things.
        """

        def set(r_model, model, lengthscale, noise_var, set_online=False):
            r_model.covar_module.base_kernel.lengthscale = lengthscale
            if set_online:
                model.covar_module.base_kernel.lengthscale = lengthscale

            r_model.likelihood.noise = noise_var
            if set_online:
                model.likelihood.noise = noise_var
            return r_model, model

        def observe(r_model, model, xs, ys):
            r_model.eval()
            r_model(xs)
            r_model = r_model.get_fantasy_model(xs, ys)

            model.eval()
            model(xs)
            model = model.get_online_model(xs, ys)
            return r_model, model

        def update(r_model, model):
            print("------------------------------------")
            r_model.train()
            r_optim = torch.optim.SGD(r_model.parameters(), self.lr)

            model.train()
            optim = torch.optim.SGD(model.parameters(), self.lr)

            with gpytorch.settings.fast_computations(), gpytorch.settings.max_cholesky_size(
                1
            ), gpytorch.settings.skip_logdet_forward():
                r_mll = ExactMarginalLogLikelihood(r_model.likelihood, r_model)
                r_train_output = r_model(r_model.train_inputs[0])
                r_mll_val = r_mll(r_train_output, r_model.train_targets)

                mll = WoodburyExactMarginalLogLikelihood(model.likelihood, model)
                train_output = model(model.train_inputs[0])
                mll_val = mll(train_output, model.train_targets)

                loss = -mll_val
                loss.backward()
                r_loss = -r_mll_val
                r_loss.backward()

            print(
                "online ls grad",
                model.covar_module.base_kernel.raw_lengthscale.grad.item(),
            )
            print(
                "ski ls grad",
                r_model.covar_module.base_kernel.raw_lengthscale.grad.item(),
            )

            print("online ls", model.covar_module.base_kernel.lengthscale.item())
            print("ski ls", r_model.covar_module.base_kernel.lengthscale.item())

            print("online noise grad", model.likelihood.raw_noise.grad.item())
            print("ski noise grad", r_model.likelihood.raw_noise.grad.item())

            print("online noise", model.likelihood.noise.item())
            print("ski noise", r_model.likelihood.noise.item())

            # Make sure the gradients are the same
            np.testing.assert_allclose(
                model.covar_module.base_kernel.raw_lengthscale.grad.item(),
                r_model.covar_module.base_kernel.raw_lengthscale.grad.item(),
                rtol=0.01,
                atol=0.01,
            )
            np.testing.assert_allclose(
                model.likelihood.raw_noise.grad.item(),
                r_model.likelihood.raw_noise.grad.item(),
                rtol=0.01,
                atol=0.01,
            )

            r_optim.step()
            r_optim.zero_grad()

            optim.step()
            optim.zero_grad()
            model.get_updated_hyper_strategy()

            # Make sure the values are the same
            np.testing.assert_allclose(
                model.covar_module.base_kernel.lengthscale.item(),
                r_model.covar_module.base_kernel.lengthscale.item(),
                rtol=0.01,
                atol=0.01,
            )
            np.testing.assert_allclose(
                model.likelihood.noise.item(),
                r_model.likelihood.noise.item(),
                rtol=0.01,
                atol=0.01,
            )

            # Verify the gradients are the same = 0
            np.testing.assert_allclose(
                model.covar_module.base_kernel.raw_lengthscale.grad.item(),
                r_model.covar_module.base_kernel.raw_lengthscale.grad.item(),
                rtol=0.01,
                atol=0.01,
            )
            np.testing.assert_allclose(
                model.likelihood.raw_noise.grad.item(),
                r_model.likelihood.raw_noise.grad.item(),
            )
            return r_model, model

        def init():
            r_lik = GaussianLikelihood()
            r_kernel = GridInterpolationKernelWithFantasy(
                RBFKernel(), grid_size=self.grid_size, grid_bounds=[(-4.0, 14.0)]
            ).double()
            r_model = RegularExactGP(self.xs, self.labels, r_lik, r_kernel, ZeroMean())

            lik = GaussianLikelihood()
            kernel = GridInterpolationKernelWithFantasy(
                RBFKernel(), grid_size=self.grid_size, grid_bounds=[(-4.0, 14.0)]
            ).double()
            model = OnlineWoodburyGP(self.xs, self.labels, lik, kernel, ZeroMean())
            return r_model, model

        # r_model, model = init()
        # set(r_model, model, self.lengthscale, self.noise_var)
        # update(r_model, model)

        r_model, model = init()
        set(r_model, model, self.lengthscale, self.noise_var, set_online=True)
        r_model, model = observe(
            r_model, model, self.points_sequence[1], self.targets_sequence[1]
        )
        r_model, model = update(r_model, model)
        set(
            r_model,
            model,
            deepcopy(model.covar_module.base_kernel.lengthscale.item()),
            deepcopy(model.likelihood.noise.item()),
        )
        r_model, model = update(r_model, model)
        set(
            r_model,
            model,
            deepcopy(model.covar_module.base_kernel.lengthscale.item()),
            deepcopy(model.likelihood.noise.item()),
        )
        update(r_model, model)

        r_model, model = init()
        set(r_model, model, self.lengthscale, self.noise_var, set_online=True)
        r_model, model = observe(
            r_model, model, self.points_sequence[1], self.targets_sequence[1]
        )
        r_model, model = observe(
            r_model, model, self.points_sequence[2], self.targets_sequence[2]
        )
        r_model, model = update(r_model, model)
        set(
            r_model,
            model,
            deepcopy(model.covar_module.base_kernel.lengthscale.item()),
            deepcopy(model.likelihood.noise.item()),
        )
        r_model, model = update(r_model, model)
        set(
            r_model,
            model,
            deepcopy(model.covar_module.base_kernel.lengthscale.item()),
            deepcopy(model.likelihood.noise.item()),
        )
        update(r_model, model)

        r_model, model = init()
        set(r_model, model, self.lengthscale, self.noise_var, set_online=True)
        r_model, model = observe(
            r_model, model, self.points_sequence[1], self.targets_sequence[1]
        )
        r_model, model = observe(
            r_model, model, self.points_sequence[2], self.targets_sequence[2]
        )
        r_model, model = observe(
            r_model, model, self.points_sequence[3], self.targets_sequence[3]
        )
        r_model, model = update(r_model, model)
        set(
            r_model,
            model,
            deepcopy(model.covar_module.base_kernel.lengthscale.item()),
            deepcopy(model.likelihood.noise.item()),
        )
        r_model, model = update(r_model, model)
        set(
            r_model,
            model,
            deepcopy(model.covar_module.base_kernel.lengthscale.item()),
            deepcopy(model.likelihood.noise.item()),
        )
        update(r_model, model)

        r_model, model = init()
        set(r_model, model, self.lengthscale, self.noise_var, set_online=True)
        r_model, model = observe(
            r_model, model, self.points_sequence[1], self.targets_sequence[1]
        )
        r_model, model = observe(
            r_model, model, self.points_sequence[2], self.targets_sequence[2]
        )
        r_model, model = observe(
            r_model, model, self.points_sequence[3], self.targets_sequence[3]
        )
        r_model, model = observe(
            r_model, model, self.points_sequence[4], self.targets_sequence[4]
        )
        r_model, model = update(r_model, model)
        set(
            r_model,
            model,
            deepcopy(model.covar_module.base_kernel.lengthscale.item()),
            deepcopy(model.likelihood.noise.item()),
        )
        r_model, model = update(r_model, model)
        set(
            r_model,
            model,
            deepcopy(model.covar_module.base_kernel.lengthscale.item()),
            deepcopy(model.likelihood.noise.item()),
        )
        update(r_model, model)

    def test_online_eval_mode(self):

        r_lik = GaussianLikelihood()
        r_lik.noise = 1.0
        # r_kernel = RBFKernel().double()
        r_kernel = GridInterpolationKernelWithFantasy(
            RBFKernel(), grid_size=self.grid_size, grid_bounds=[(-4.0, 14.0)]
        ).double()
        r_model = RegularExactGP(
            self.xs, self.labels, r_lik, r_kernel, deepcopy(self.mean_module)
        )
        r_model.eval()
        r_model(self.new_points).mean
        r_model = r_model.get_fantasy_model(self.new_points, self.new_targets)

        r_test_output = r_lik(r_model(self.test_points))

        lik = GaussianLikelihood()
        lik.noise = 1.0
        model = OnlineWoodburyGP(self.xs, self.labels, lik, self.kernel, self.mean_module)
        model.eval()
        model(self.new_points).mean
        model = model.get_fantasy_model(self.new_points, self.new_targets)

        test_output = lik(model(self.test_points))

        np.testing.assert_allclose(
            r_test_output.mean.detach().numpy(), test_output.mean.detach().numpy()
        )

        np.testing.assert_allclose(
            r_test_output.covariance_matrix.detach().numpy(),
            test_output.covariance_matrix.detach().numpy(),
        )


# class WoodburyOnlineTestCase(TestCase, BaseTestCaseMixin):
#     def setUp(self):
#         super()._setUp()


class ShermanMorrisonOnlineTestCase(TestCase, BaseTestCaseMixin):
    def setUp(self):
        super()._setUp()

    def test_train_mll_backprop(self):
        super().test_train_mll_backprop()

    def test_eval_mode(self):
        super().test_eval_mode()

    def test_online_train_mll_backprop(self):
        super().test_online_train_mll_backprop()

    def test_mixed_online_train_mll_backprop(self):
        super().test_mixed_online_train_mll_backprop()

    def test_online_eval_mode(self):
        super().test_online_eval_mode()


if __name__ == "__main__":
    main()

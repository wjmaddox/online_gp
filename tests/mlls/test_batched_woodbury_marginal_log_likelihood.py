import torch
import numpy as np
from unittest import TestCase, main
import gpytorch
from copy import deepcopy
from botorch.models import SingleTaskGP, FixedNoiseGP
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood

from online_gp.models.batched_fixed_noise_online_gp import FixedNoiseOnlineSKIGP
from online_gp.mlls.batched_woodbury_marginal_log_likelihood import BatchedWoodburyMarginalLogLikelihood

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

class TestBatchedWoodburyMarginalLogLikelihood(TestCase):
    def setUp(self, batched=False, learnable=False):
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.random.manual_seed(10)

        train_x = torch.rand(10, 2)
        train_y = torch.sin(2 * train_x[:, 0] + 3 * train_x[:, 1]).unsqueeze(-1)
        train_y_var = 0.1 * torch.ones_like(train_y)
        if batched:
            train_y = torch.cat(
                (
                    train_y, 
                    train_y + 0.3 * torch.randn_like(train_y),
                    train_y + 0.3 * torch.randn_like(train_y),
                ),
                dim=1
            )
            train_y_var = train_y_var.repeat(1, 3)

        model = FixedNoiseOnlineSKIGP(
            train_inputs=train_x,
            train_targets=train_y,
            train_noise_term=train_y_var,
            grid_bounds=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            grid_size=5,
            learn_additional_noise=learnable
        )
        equivalent_model = SingleTaskGP(
            train_X=train_x, 
            train_Y=train_y, 
            likelihood=FixedNoiseGaussianLikelihood(train_y_var.t(), learn_additional_noise=learnable),
            covar_module = deepcopy(model.covar_module)
        )
        equivalent_model.mean_module = ZeroMean()

        return model, equivalent_model, train_x, train_y

    def _test_mll_computation(self, model, emodel, train_x, train_y):
        print(model.likelihood)
        mll = BatchedWoodburyMarginalLogLikelihood(model.likelihood, model)
        emll = ExactMarginalLogLikelihood(emodel.likelihood, emodel)

        with gpytorch.settings.skip_logdet_forward(False):
            loss1 = mll(model(train_x), train_y)

            loss2 = emll(emodel(*emodel.train_inputs), emodel.train_targets)
            print('losses: ', loss1, loss2)
            loss1.sum().backward()
            loss2.sum().backward()

            model_grad = flatten([p.grad for p in model.parameters()])
            emodel_grad = flatten([p.grad for p in emodel.parameters()])
            # print('gradients: ', model_grad, emodel_grad)

            self.assertTrue(torch.allclose(loss1, loss2))
            self.assertTrue(torch.allclose(model_grad, emodel_grad))


    def test_fixed_noise(self):
        args = self.setUp(batched=False, learnable=False)
        self._test_mll_computation(*args)

    def test_batched_fixed_noise(self):
        args = self.setUp(batched=True, learnable=False)
        self._test_mll_computation(*args)

    # def test_learnable_noise(self):
    #     args = self.setUp(learnable=True)
    #     self._test_mll_computation(*args)        

if __name__ == "__main__":
    main()

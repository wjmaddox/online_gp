import torch
import gpytorch

from gpytorch.mlls import ExactMarginalLogLikelihood
from online_gp.models.gp_dirichlet_classification import DirichletGPClassifier
from botorch.models.gp_regression import FixedNoiseGP
from gpytorch.kernels import RBFKernel, ScaleKernel


class OnlineExactClassifier(DirichletGPClassifier):
    def __init__(self, stem, init_x, init_y, alpha_eps, lr, **kwargs):
        stem = stem.to(init_x.device)
        transformed_y, _, sigma2_i = self._transform_targets(init_y, alpha_eps)
        if transformed_y.t().shape[0] != 1:
            _batch_shape = transformed_y.t().shape[:-1]
        else:
            _batch_shape = torch.Size()
        features = stem(init_x)
        gp = FixedNoiseGP(
            features,
            transformed_y,
            sigma2_i,
            covar_module=ScaleKernel(RBFKernel(batch_shape=_batch_shape, ard_num_dims=stem.output_dim),
                                     batch_shape=_batch_shape)
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        super().__init__(stem, gp, mll, alpha_eps, lr)
        self._raw_inputs = [init_x]
        self._target_batch_shape = _batch_shape

    def fit(self, inputs, targets, num_epochs, test_dataset=None):
        records = []
        transformed_targets, _, _ = self._transform_targets(targets, self.alpha_eps)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, num_epochs, 1e-4)
        for epoch in range(num_epochs):
            self.train()
            self.optimizer.zero_grad()
            self.gp.zero_grad()
            features = self.stem(inputs)
            self.set_train_data(features, self.gp.train_targets)
            train_dist = self.gp(*self.gp.train_inputs)
            with gpytorch.settings.skip_logdet_forward(True):
                loss = -mll(train_dist, transformed_targets.t()).sum()
            loss.backward()
            self.optimizer.step()
            lr_scheduler.step()

            test_acc = float('NaN')
            if test_dataset is not None:
                test_x, test_y = test_dataset[:]
                test_pred = self.predict(test_x)
                test_acc = test_pred.eq(test_y).float().mean().item()
            records.append({'train_loss': loss.item(), 'test_acc': test_acc,
                            'epoch': epoch + 1})
        self.eval()
        return records

    def update(self, inputs, targets, update_stem=True, update_gp=True):
        inputs = inputs.view(-1, self.stem.input_dim)
        targets = targets.view(-1)

        # add latest observation
        transformed_targets, _, sigmai_2 = self._transform_targets(targets, self.alpha_eps)
        with torch.no_grad():
            features = self.stem(inputs)
        self.gp = self.gp.condition_on_observations(features, transformed_targets, noise=sigmai_2)

        # update GP training data
        self.train()
        self._raw_inputs = [torch.cat([*self._raw_inputs, inputs])]
        features = self.stem(*self._raw_inputs)  # make sure batchnorm stats are getting updated
        features = features if update_stem else features.detach()
        self.set_train_data(features, self.gp.train_targets)

        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)

        # update stem and GP
        if update_gp:
            self.optimizer.zero_grad()
            self.gp.zero_grad()
            with gpytorch.settings.skip_logdet_forward(True):
                train_dist = self.gp(*self.gp.train_inputs)
                loss = -self.mll(train_dist, self.gp.train_targets).sum()
            loss.backward()
            self.optimizer.step()

        # update GP training data again
        features = self.stem(*self._raw_inputs)
        self.set_train_data(features, self.gp.train_targets)

        self.eval()
        stem_loss = gp_loss = loss.item() if update_gp else 0.
        return stem_loss, gp_loss

    def set_train_data(self, inputs, targets):
        inputs = inputs.expand(*self._target_batch_shape, -1, -1)
        super().set_train_data(inputs, targets, noise=None)

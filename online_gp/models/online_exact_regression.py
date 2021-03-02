import torch
import gpytorch

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from online_gp.utils import regression


class OnlineExactRegression(torch.nn.Module):
    def __init__(self, stem, init_x, init_y, lr, **kwargs):
        super().__init__()
        self.stem = stem.to(init_x.device)
        if init_y.t().shape[0] != 1:
            _batch_shape = init_y.t().shape[:-1]
        else:
            _batch_shape = torch.Size()
        features = self.stem(init_x)
        self.gp = SingleTaskGP(
            features,
            init_y,
            covar_module=ScaleKernel(RBFKernel(batch_shape=_batch_shape, ard_num_dims=stem.output_dim),
                                     batch_shape=_batch_shape)
        )
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self._raw_inputs = [init_x]
        self._target_batch_shape = _batch_shape
        self.target_dim = init_y.size(-1)

    def update(self, inputs, targets, update_stem=True, update_gp=True):
        inputs = inputs.view(-1, self.stem.input_dim)
        targets = targets.view(-1, self.target_dim)

        # add observation
        self.train()
        self._raw_inputs = [torch.cat([*self._raw_inputs, inputs])]
        self.gp.train_targets = torch.cat([
            self.gp.train_targets,
            self._reshape_targets(targets)
        ], dim=-1)

        if update_stem:
            self._refresh_features(*self._raw_inputs, strict=False)
        else:
            with torch.no_grad():
                self._refresh_features(*self._raw_inputs, strict=False)

        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        # update stem and GP
        if update_gp:
            self.optimizer.zero_grad()
            with gpytorch.settings.skip_logdet_forward(True):
                train_dist = self.gp(*self.gp.train_inputs)
                loss = -self.mll(train_dist, self.gp.train_targets).sum()
            loss.backward()
            self.optimizer.step()
            self.gp.zero_grad()

        # update GP training data again
        if update_stem:
            with torch.no_grad():
                self._refresh_features(*self._raw_inputs)

        self.eval()
        stem_loss = gp_loss = loss.item() if update_gp else 0.
        return stem_loss, gp_loss

    def fit(self, inputs, targets, num_epochs, test_dataset=None):
        records = []
        self.gp.train_targets = self._reshape_targets(targets)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, num_epochs, 1e-4)
        for epoch in range(num_epochs):
            self.train()
            self.mll.train()
            self.optimizer.zero_grad()
            self._refresh_features(inputs)
            train_dist = self.gp(*self.gp.train_inputs)
            with gpytorch.settings.skip_logdet_forward(False):
                loss = -self.mll(train_dist, self.gp.train_targets).sum()
            loss.backward()
            self.optimizer.step()
            lr_scheduler.step()
            self.gp.zero_grad()

            rmse = nll = float('NaN')
            if test_dataset is not None:
                test_x, test_y = test_dataset[:]
                rmse, nll = self.evaluate(test_x, test_y)
            records.append({'train_loss': loss.item(), 'test_rmse': rmse,
                            'test_nll': nll, 'noise': self.gp.likelihood.noise.mean().item(),
                            'epoch': epoch + 1})

        with torch.no_grad():
            self._refresh_features(inputs)

        self.eval()
        return records

    def forward(self, inputs):
        inputs = inputs.view(-1, self.stem.input_dim)
        features = self.stem(inputs)
        return self.gp(features)

    def predict(self, inputs):
        self.eval()
        pred_dist = self(inputs)
        pred_dist = self.gp.likelihood(pred_dist)
        return pred_dist.mean, pred_dist.variance

    def evaluate(self, inputs, targets):
        inputs = inputs.view(-1, self.stem.input_dim)
        targets = targets.view(-1, self.target_dim)
        with torch.no_grad():
            return regression.evaluate(self, inputs, targets)

    def set_train_data(self, inputs, targets, strict):
        inputs = inputs.expand(*self._target_batch_shape, -1, -1)
        if self.target_dim == 1:
            targets = targets.squeeze(0)
        self.gp.set_train_data(inputs, targets, strict)

    def _reshape_targets(self, targets):
        targets = targets.view(-1, self.target_dim)
        if targets.size(-1) == 1:
            targets = targets.squeeze(-1)
        else:
            targets = targets.t()
        return targets

    def _refresh_features(self, inputs, strict=True):
        features = self.stem(inputs)
        self.set_train_data(features, self.gp.train_targets, strict)
        return features

    def set_lr(self, gp_lr, stem_lr=None, bn_mom=None):
        stem_lr = gp_lr if stem_lr is None else stem_lr
        self.optimizer = torch.optim.Adam([
            dict(params=self.gp.parameters(), lr=gp_lr),
            dict(params=self.stem.parameters(), lr=stem_lr)
        ])
        if bn_mom is not None:
            for m in self.stem.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    m.momentum = bn_mom

    @property
    def noise(self):
        return self.gp.likelihood.noise
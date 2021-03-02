from .variational_gp_model import VariationalGPModel
from gpytorch.likelihoods import GaussianLikelihood
import torch
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.mlls import VariationalELBO
from online_gp.utils import regression
import math


class OnlineSVGPRegression(torch.nn.Module):
    def __init__(
            self,
            stem,
            init_x,
            init_y,
            num_inducing,
            lr,
            streaming=False,
            prior_beta=1.,
            online_beta=1.,
            learn_inducing_locations=True,
            num_update_steps=1,
            covar_module=None,
            inducing_points=None,
            **kwargs
        ):
        super().__init__()
        assert init_y.ndimension() == 2
        target_dim = init_y.size(-1)
        batch_shape = target_dim if target_dim > 1 else []
        likelihood = GaussianLikelihood(batch_shape=batch_shape)
        if inducing_points is None:
            inducing_points = torch.empty(num_inducing, stem.output_dim)
            inducing_points.uniform_(-1, 1)
        mean_module = ZeroMean()
        if covar_module is None:
            covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=stem.output_dim, batch_shape=batch_shape),
                batch_shape=batch_shape
            )
        self.gp = VariationalGPModel(inducing_points, mean_module, covar_module, streaming, likelihood,
                                     beta=online_beta, learn_inducing_locations=learn_inducing_locations)
        self.mll = None
        self.stem = stem
        self.optimizer = torch.optim.Adam(self.param_groups(lr))
        self.num_update_steps = num_update_steps
        self._raw_inputs = [init_x]
        self.target_dim = target_dim
        self._prior_beta = prior_beta

    def forward(self, inputs):
        inputs = inputs.view(-1, self.stem.input_dim)
        features = self.stem(inputs)
        return self.gp(features)

    def fit(self, inputs, targets, num_epochs, test_dataset=None):
        streaming_state = self.gp.streaming
        self.gp.set_streaming(False)
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
        num_batches = len(dataloader)

        self.mll = VariationalELBO(self.gp.likelihood, self.gp, num_data=len(dataset), beta=1.0)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, num_epochs, 1e-4)
        records = []
        for epoch in range(num_epochs):
            self.train()
            avg_loss = 0
            for input_batch, target_batch in dataloader:
                self.optimizer.zero_grad()
                train_dist = self(input_batch)
                target_batch = self._reshape_targets(target_batch)
                loss = -self.mll(train_dist, target_batch).mean()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item() / num_batches
            lr_scheduler.step()

            rmse = nll = float('NaN')
            if test_dataset is not None:
                test_x, test_y = test_dataset[:]
                with torch.no_grad():
                    rmse, nll = self.evaluate(test_x, test_y)

            records.append(dict(train_loss=avg_loss, test_rmse=rmse, test_nll=nll,
                                noise=self.gp.likelihood.noise.mean().item(), epoch=epoch + 1))

        self.gp.set_streaming(streaming_state)
        self.eval()
        return records

    def predict(self, inputs):
        self.eval()
        pred_dist = self(inputs)
        pred_dist = self.gp.likelihood(pred_dist)
        return pred_dist.mean, pred_dist.variance

    def evaluate(self, inputs, targets):
        inputs = inputs.view(-1, self.stem.input_dim)
        targets = targets.view(-1, self.target_dim)
        with torch.no_grad():
            rmse, nll = regression.evaluate(self, inputs, targets)
        return rmse, nll

    def update(self, inputs, targets, update_stem=True):
        if self.gp.streaming:
            self.gp.register_streaming_loss()
        inputs = inputs.view(-1, self.stem.input_dim)
        targets = targets.view(-1, self.target_dim)
        self.mll = VariationalELBO(self.gp.likelihood, self.gp, num_data=inputs.size(0),
                                   beta=self._prior_beta)
        self.train()
        for _ in range(self.num_update_steps):
            self.optimizer.zero_grad()
            features = self._get_features(inputs)
            features = features if update_stem else features.detach()
            train_dist = self.gp(features)
            targets = self._reshape_targets(targets)
            loss = -self.mll(train_dist, targets).mean()
            loss.backward()
            self.optimizer.step()

        self.eval()
        self._raw_inputs = [torch.cat([*self._raw_inputs, inputs])]
        stem_loss = gp_loss = loss.item()
        return stem_loss, gp_loss

    def set_lr(self, gp_lr, stem_lr=None, bn_mom=None):
        # stem_lr = gp_lr if stem_lr is None else stem_lr
        # self.optimizer = torch.optim.Adam([
        #     dict(params=self.gp.parameters(), lr=gp_lr),
        #     dict(params=self.stem.parameters(), lr=stem_lr)
        # ])
        # if bn_mom is not None:
        #     for m in self.stem.modules():
        #         if isinstance(m, torch.nn.BatchNorm1d):
        #             m.momentum = bn_mom
        self.optimizer = torch.optim.Adam(self.param_groups(gp_lr))

    def param_groups(self, base_lr):
        hyper_group = dict(params=[], lr=base_lr)
        variational_group = dict(params=[], lr=base_lr / 10)
        stem_group = dict(params=self.stem.parameters(), lr=base_lr / 10)

        for name, param in self.gp.named_parameters():
            if 'variational' in name:
                variational_group['params'].append(param)
            else:
                hyper_group['params'].append(param)
        return hyper_group, variational_group, stem_group

    def _get_features(self, inputs):
        # update batch norm stats
        inputs = inputs.view(-1, self.stem.input_dim)
        num_inputs = inputs.size(0)
        num_seen = self._raw_inputs[0].size(0)
        batch_size = 1024
        batch_idxs = torch.randint(0, num_seen, (batch_size,))
        input_batch = self._raw_inputs[0][batch_idxs]
        input_batch = torch.cat([inputs, input_batch])
        features = self.stem(input_batch)
        return features[:num_inputs]

    def _reshape_targets(self, targets):
        targets = targets.view(-1, self.target_dim)
        if targets.size(-1) == 1:
            targets = targets.squeeze(-1)
        else:
            targets = targets.t()
        return targets

    @property
    def noise(self):
        return self.gp.likelihood.noise
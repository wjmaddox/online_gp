from .streaming_sgpr import StreamingSGPR, StreamingSGPRBound
import torch
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.mlls import VariationalELBO
from online_gp.utils import regression, cuda
import math


class OnlineSGPRegression(torch.nn.Module):
    def __init__(
            self,
            stem,
            init_x,
            init_y,
            num_inducing,
            lr,
            learn_inducing_locations=True,
            num_update_steps=1,
            covar_module=None,
            inducing_points=None,
            jitter=1e-4,
            **kwargs
        ):
        super().__init__()
        assert init_y.ndimension() == 2
        target_dim = init_y.size(-1)
        batch_shape = target_dim if target_dim > 1 else []

        if inducing_points is None:
            inducing_points = torch.empty(num_inducing, stem.output_dim)
            inducing_points.uniform_(-1, 1)

        if covar_module is None:
            covar_module = ScaleKernel(
                RBFKernel(ard_num_dims=stem.output_dim, batch_shape=batch_shape),
                batch_shape=batch_shape
            )
        self.gp = StreamingSGPR(inducing_points, learn_inducing_locations=learn_inducing_locations,
                                covar_module=covar_module, num_data=init_x.size(-2), jitter=jitter)
        self.mll = None
        self.stem = stem
        self.optimizer = None
        self.num_update_steps = num_update_steps
        self._raw_inputs = [init_x]
        self.target_dim = target_dim

    def forward(self, inputs):
        inputs = inputs.view(-1, self.stem.input_dim)
        features = self.stem(inputs)
        return self.gp(features)

    def fit(self, inputs, targets, num_epochs, test_dataset=None):
        # elbo = VariationalELBO(self.gp.likelihood, self.gp, num_data=inputs.size(-2), beta=1.0)
        elbo = StreamingSGPRBound(self.gp)
        trainable_params = [
            dict(params=self.gp.likelihood.parameters(), lr=1e-1),
            dict(params=self.gp.covar_module.parameters(), lr=1e-1),
            dict(params=self.gp.variational_strategy.inducing_points, lr=1e-2),
            dict(params=self.gp.variational_strategy._variational_distribution.parameters(), lr=1e-2),
            dict(params=self.stem.parameters(), lr=1e-2)
        ]
        optimizer = torch.optim.Adam(trainable_params)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 1e-4)

        records = []
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            features = self.stem(inputs)
            loss = -elbo(features, targets).sum()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            rmse = nll = float('NaN')
            if test_dataset is not None:
                self.eval()
                test_x, test_y = test_dataset[:]
                with torch.no_grad():
                    features = self.stem(inputs)
                    self.gp.update_variational_distribution(features, targets)
                    rmse, nll = self.evaluate(test_x, test_y)

            records.append(dict(train_loss=loss.item(), test_rmse=rmse, test_nll=nll,
                                noise=self.gp.likelihood.noise.mean().item(), epoch=epoch + 1))

        self.eval()
        features = self.stem(inputs)
        self.gp = cuda.try_cuda(self.gp.get_fantasy_model(features, targets, resample_ratio=0))
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
        self.train()
        inputs = inputs.view(-1, self.stem.input_dim)
        targets = targets.view(-1, self.target_dim)
        elbo = StreamingSGPRBound(self.gp, combine_terms=False)
        for _ in range(self.num_update_steps):
            self.optimizer.zero_grad()
            features = self._get_features(inputs)
            features = features if update_stem else features.detach()
            logp, trace, _, _ = elbo(features, targets)
            loss = -(logp + trace).sum()
            # loss = -logp.sum()
            # print(f'[O-SGPR] Streaming ELBO: {loss.item():0.4f}')
            loss.backward()
            self.optimizer.step()

        self.eval()
        features = self.stem(inputs)
        # resample_ratio = 1 / self.gp.variational_strategy.inducing_points.size(-2)
        resample_ratio = 0
        self.gp = cuda.try_cuda(self.gp.get_fantasy_model(features, targets, resample_ratio))
        self._raw_inputs = [torch.cat([*self._raw_inputs, inputs])]
        stem_loss = gp_loss = loss.item()
        return stem_loss, gp_loss

    def set_lr(self, gp_lr, stem_lr=None, bn_mom=None):
        stem_lr = gp_lr / 10 if stem_lr is None else stem_lr
        trainable_params = [
            dict(params=self.gp.likelihood.parameters(), lr=gp_lr),
            dict(params=self.gp.covar_module.parameters(), lr=gp_lr),
            dict(params=self.gp.variational_strategy.inducing_points, lr=gp_lr / 10),
            dict(params=self.stem.parameters(), lr=stem_lr)
        ]
        self.optimizer = torch.optim.Adam(trainable_params)

    # def param_groups(self, base_lr):
    #     hyper_group = dict(params=[], lr=base_lr)
    #     variational_group = dict(params=[], lr=base_lr / 10)
    #     stem_group = dict(params=self.stem.parameters(), lr=base_lr / 10)
    #
    #     for name, param in self.gp.named_parameters():
    #         if 'variational' in name:
    #             variational_group['params'].append(param)
    #         else:
    #             hyper_group['params'].append(param)
    #     return hyper_group, variational_group, stem_group

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
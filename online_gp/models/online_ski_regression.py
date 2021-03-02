import torch
import gpytorch

from gpytorch.mlls import ExactMarginalLogLikelihood
from online_gp.mlls.batched_woodbury_marginal_log_likelihood import BatchedWoodburyMarginalLogLikelihood
from online_gp.models.batched_fixed_noise_online_gp import FixedNoiseOnlineSKIGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from online_gp.models.stems import Identity
from online_gp.mlls.streaming_partial_mll import sm_partial_mll
from online_gp.utils import regression
from torch.optim.lr_scheduler import CosineAnnealingLR
from online_gp.settings import detach_interp_coeff


class OnlineSKIRegression(torch.nn.Module):
    def __init__(self, stem, init_x, init_y, lr, grid_size, grid_bound, covar_module=None, **kwargs):
        super().__init__()
        self.stem = stem.to(init_x.device)
        assert init_y.ndim == 2, "targets must have explicit output dimension"
        if init_y.size(-1) == 1:
            target_batch_shape = []
        else:
            target_batch_shape = torch.Size([init_y.size(-1)])
        features = self.stem(init_x).detach()
        noise_term = torch.ones_like(init_y)
        grid_bound += 1e-1
        self.gp = FixedNoiseOnlineSKIGP(
            features,
            init_y,
            noise_term,
            covar_module=covar_module,
            grid_bounds=torch.tensor([[-grid_bound, grid_bound]] * stem.output_dim),
            grid_size=[grid_size] * stem.output_dim,
            learn_additional_noise=True
        )
        self.mll = BatchedWoodburyMarginalLogLikelihood(self.gp.likelihood, self.gp)
        self.gp_optimizer = torch.optim.Adam(self.gp.parameters(), lr=lr)
        self.stem_optimizer = torch.optim.Adam(self.stem.parameters(), lr=lr)
        self._target_batch_shape = target_batch_shape
        self.target_dim = init_y.size(-1)
        self._raw_inputs = [init_x]

    def forward(self, inputs):
        inputs = inputs.view(-1, self.stem.input_dim)
        features = self.stem(inputs)
        return self.gp(features)

    def _reshape_targets(self, targets):
        targets = targets.view(-1, self.target_dim)
        if targets.size(-1) == 1:
            targets = targets.squeeze(-1)
        else:
            targets = targets.t()
        return targets

    def predict(self, inputs):
        self.eval()
        pred_dist = self(inputs)
        pred_mean = pred_dist.mean.view(-1, self.target_dim)
        pred_var = pred_dist.variance.view(-1, self.target_dim)
        pred_var = pred_var + self.gp.likelihood.second_noise
        return pred_mean, pred_var

    def evaluate(self, inputs, targets):
        inputs = inputs.view(-1, self.stem.input_dim)
        targets = targets.view(-1, self.target_dim)
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024)
        # Don't use `torch.no_grad` here, caches will be used for training
        self.eval()
        rmse, nll = 0, 0
        num_batches = len(dataloader)
        for input_batch, target_batch in dataloader:
            pred_mean, pred_var = self.predict(input_batch)
            rmse += (pred_mean - target_batch).pow(2).mean().sqrt().item() / num_batches
            diag_dist = torch.distributions.Normal(pred_mean, pred_var.sqrt())
            nll += -diag_dist.log_prob(target_batch).mean().item() / num_batches
        return rmse, nll

    def fit(self, inputs, targets, num_epochs, test_dataset=None):
        records = []
        gp_lr_sched = CosineAnnealingLR(self.gp_optimizer, num_epochs, 1e-4)
        stem_lr_sched = CosineAnnealingLR(self.stem_optimizer, num_epochs, 1e-4)
        features = self._refresh_features(inputs, targets)
        for epoch in range(num_epochs):
            self.train()
            self.mll.train()
            self.stem_optimizer.zero_grad()
            self.gp_optimizer.zero_grad()
            train_dist = self.gp(features)
            loss = -self.mll(train_dist, targets).sum()
            loss.backward()
            self.stem_optimizer.step()
            self.gp_optimizer.step()
            stem_lr_sched.step()
            gp_lr_sched.step()
            features = self._refresh_features(inputs, targets)

            rmse = nll = float('NaN')
            if test_dataset is not None:
                test_x, test_y = test_dataset[:]
                rmse, nll = self.evaluate(test_x, test_y)
            records.append({'epoch': epoch + 1, 'train_loss': loss.item(),
                            'test_rmse': rmse, 'test_nll': nll,
                            'noise': self.gp.likelihood.second_noise_covar.noise.mean().item()})

        with detach_interp_coeff(True):
            self._refresh_features(inputs, targets)

        self.eval()
        return records

    def update(self, inputs, targets, update_stem=True, update_gp=True):
        inputs = inputs.view(-1, self.stem.input_dim)
        targets = targets.view(-1, self.target_dim)

        stem_loss = self._update_stem(inputs, targets) if update_stem else 0.
        gp_loss = self._update_gp(inputs, targets) if update_gp else 0.

        with torch.no_grad():
            features = self.stem(inputs)
            noise_term = torch.ones_like(targets)
            self.gp.condition_on_observations(features, targets, noise_term, inplace=True)
            self._raw_inputs = [torch.cat([*self._raw_inputs, inputs])]
            self.stem.train()
            if update_stem:
                self._get_features(inputs)

        self.eval()
        return stem_loss, gp_loss

    def _update_gp(self, inputs, targets):
        self.gp_optimizer.zero_grad()

        self.gp.train()
        self.mll.train()
        with gpytorch.settings.skip_logdet_forward(True):
            features = self.stem(inputs)
            train_dist = self.gp(features.detach())
            loss = -self.mll(train_dist, targets).sum()
        loss.backward()
        self.gp_optimizer.step()

        self.gp.zero_grad()
        self.gp.eval()
        return loss.item()

    def _update_stem(self, inputs, targets):
        self.stem_optimizer.zero_grad()
        num_seen = self.gp.num_data

        self.stem.eval()  # we want deterministic features, so BatchNorm should be in eval mode
        new_features = self.stem(inputs)
        if new_features.requires_grad is False:
            return 0

        targets = targets.transpose(-1, -2).unsqueeze(-1)
        loss = -sm_partial_mll(self.gp, new_features, targets, num_seen).sum()
        loss.backward()
        self.stem_optimizer.step()

        return loss.item()

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

    def _refresh_features(self, inputs, targets):
        features = self.stem(inputs)
        self.set_train_data(features, targets)
        self.gp.zero_grad()  # dump W-related caches
        # self.gp.init_kernel_cache()  # refresh W'y
        return features

    def set_train_data(self, inputs, targets):
        noise = torch.ones_like(targets)
        self.gp.set_train_data(inputs, targets, noise)

    def set_lr(self, gp_lr, stem_lr=None, bn_mom=None):
        stem_lr = gp_lr if stem_lr is None else stem_lr
        self.gp_optimizer = torch.optim.Adam(self.gp.parameters(), lr=gp_lr)
        self.stem_optimizer = torch.optim.Adam(self.stem.parameters(), lr=stem_lr)
        if bn_mom is not None:
            for m in self.stem.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    m.momentum = bn_mom

    @property
    def noise(self):
        return self.gp.likelihood.noise
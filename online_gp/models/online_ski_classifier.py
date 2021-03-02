import torch
import gpytorch

from online_gp.models.batched_fixed_noise_online_gp import FixedNoiseOnlineSKIGP, _get_wmat_from_kernel
from online_gp.mlls.batched_woodbury_marginal_log_likelihood import BatchedWoodburyMarginalLogLikelihood
from online_gp.models.gp_dirichlet_classification import DirichletGPClassifier
from online_gp.models.stems import Identity, LinearStem
from online_gp.mlls.streaming_partial_mll import sm_partial_mll
from online_gp.settings import detach_interp_coeff
from torch.optim.lr_scheduler import CosineAnnealingLR


class OnlineSKIClassifier(DirichletGPClassifier):
    def __init__(self, stem, init_x, init_y, alpha_eps, lr, grid_size, grid_bound, **kwargs):
        stem = stem.to(init_x.device)
        transformed_y, _, sigma2_i = self._transform_targets(init_y, alpha_eps)
        if transformed_y.t().shape[0] != 1:
            _batch_shape = transformed_y.t().shape[:-1]
        else:
            _batch_shape = torch.Size()

        features = stem(init_x).detach()
        gp = FixedNoiseOnlineSKIGP(
            features,
            transformed_y,
            sigma2_i,
            grid_bounds=torch.tensor([[-grid_bound, grid_bound]] * stem.output_dim),
            grid_size=[grid_size] * stem.output_dim
        )
        mll = BatchedWoodburyMarginalLogLikelihood(gp.likelihood, gp)
        super().__init__(stem, gp, mll, alpha_eps, lr)
        del self.optimizer
        self.gp_optimizer = torch.optim.Adam(self.gp.parameters(), lr=lr)
        self.stem_optimizer = torch.optim.Adam(self.stem.parameters(), lr=lr)
        self._target_batch_shape = _batch_shape
        self._raw_inputs = [init_x]

    def fit(self, inputs, targets, num_epochs, test_dataset=None):
        records = []
        gp_lr_sched = CosineAnnealingLR(self.gp_optimizer, num_epochs, 1e-4)
        stem_lr_sched = CosineAnnealingLR(self.stem_optimizer, num_epochs, 1e-4)
        features = self._refresh_features(inputs, targets)
        for epoch in range(num_epochs):
            self.train()
            self.mll.train()
            self.gp_optimizer.zero_grad()
            self.stem_optimizer.zero_grad()
            train_dist = self.gp(features)
            loss = -self.mll(train_dist, targets).sum()
            loss.backward()
            self.gp_optimizer.step()
            self.stem_optimizer.step()
            gp_lr_sched.step()
            stem_lr_sched.step()
            features = self._refresh_features(inputs, targets)

            test_acc = float('NaN')
            if test_dataset is not None:
                test_x, test_y = test_dataset[:]
                test_pred = self.predict(test_x)
                test_acc = test_pred.eq(test_y).float().mean().item()
            records.append({'train_loss': loss.item(), 'test_acc': test_acc,
                            'epoch': epoch + 1})

        with detach_interp_coeff(True):
            self._refresh_features(inputs, targets)

        self.eval()
        return records

    def update(self, inputs, targets, update_stem=True, update_gp=True):
        inputs = inputs.view(-1, self.stem.input_dim)
        targets = targets.view(-1)

        targets, _, noise_term = self._transform_targets(targets, self.alpha_eps)
        stem_loss = self._update_stem(inputs, targets, noise_term) if update_stem else 0.
        gp_loss = self._update_gp(inputs, targets) if update_gp else 0.

        with torch.no_grad():
            features = self.stem(inputs)
            self.gp.condition_on_observations(features, targets, noise_term, inplace=True)
            self._raw_inputs = [torch.cat([*self._raw_inputs, inputs])]
            if update_stem:
                self.stem.train()
                self._get_features(inputs)  # update batchnorm stats

        self.eval()
        return stem_loss, gp_loss

    def _update_gp(self, inputs, targets):
        self.gp_optimizer.zero_grad()
        self.mll.train()
        self.gp.train()
        with gpytorch.settings.skip_logdet_forward(True):
            train_dist = self.gp(inputs)
            loss = -self.mll(train_dist, targets).sum()
        loss.backward()
        self.gp_optimizer.step()
        self.gp.zero_grad()
        self.gp.eval()
        return loss.item()

    def _update_stem(self, inputs, targets, noise):
        self.stem_optimizer.zero_grad()

        num_seen = self.gp.num_data

        new_features = self.stem(inputs)
        if new_features.requires_grad is False:
            return 0

        new_y = (targets / noise).t().unsqueeze(-1)

        loss = -sm_partial_mll(self.gp, new_features, new_y, num_seen).sum()
        loss.backward()
        self.stem_optimizer.step()
        return loss.item()

    def _get_features(self, inputs):
        """sample minibatch to update stem batch-norm stats"""
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
        transformed_y, _, sigma2 = self._transform_targets(targets, self.alpha_eps)
        self.set_train_data(features, transformed_y, sigma2)
        self.gp.zero_grad()
        # self.gp.init_kernel_cache()  # refresh W'y
        return features

    def set_lr(self, gp_lr, stem_lr=None, bn_mom=None):
        stem_lr = gp_lr if stem_lr is None else stem_lr
        self.gp_optimizer = torch.optim.Adam(self.gp.parameters(), lr=gp_lr)
        self.stem_optimizer = torch.optim.Adam(self.stem.parameters(), lr=stem_lr)
        if bn_mom is not None:
            for m in self.stem.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    m.momentum = bn_mom
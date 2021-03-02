import torch
import gpytorch

from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.models import IndependentModelList
from torch.distributions import Categorical, MixtureSameFamily, Normal
from gpytorch import mlls
import math


class LocalGPModel(IndependentModelList):
    def __init__(self, stem, init_x, init_y, lr, max_data_per_model, share_covar=True, **kwargs):
        stem = stem.to(init_x.device)
        if init_y.t().shape[0] != 1:
            _batch_shape = init_y.t().shape[:-1]
        else:
            _batch_shape = torch.Size()
        features = stem(init_x)

        covar_module = ScaleKernel(
            RBFKernel(
                batch_shape=_batch_shape, 
                ard_num_dims=stem.output_dim
            ),
            batch_shape=_batch_shape,
        )

        if init_x.shape[-2] < max_data_per_model:
            model_assignments = torch.zeros(init_x.shape[-2]).to(init_x.device)
            model_list = torch.nn.ModuleList([SingleTaskGP(features, init_y, covar_module=covar_module)])
        else:
            num_models = math.ceil(init_x.shape[-2] / max_data_per_model)
            model_assignments = torch.randint(num_models, (init_x.shape[-2],)).to(init_x.device)
            model_list = torch.nn.ModuleList([])
            for i in range(num_models):
                idx = (model_assignments == i)
                model_list.append(SingleTaskGP(features[idx], init_y[idx], covar_module=covar_module))
        
        super().__init__(*model_list)
        self.stem = stem
        self.covar_module = covar_module
        self.update_model_caches()
        self.max_data_per_model = max_data_per_model
        self._raw_inputs = [init_x]
        self._raw_targets = init_y
        self.input_dim = init_x.size(-1)
        self.target_dim = init_y.size(-1)
        self._assignments = model_assignments
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.mll = mlls.SumMarginalLogLikelihood(self.models[0].likelihood, self)
        
    def _construct_weights(self, X):
        weights = torch.stack([self.covar_module(X, cm).evaluate() for cm in self.centers])
        return weights
    
    def __call__(self, X, *args, **kwargs):
        if self.training:
            # at train time we need to ensure that each model only receives its own training inputs
            return [model(*model.train_inputs) for model in self.models]
        else:
            # now we compute weights and scale the posteriors by the weights
            # returning a mixture distribution
            weights = self._construct_weights(X)[...,0].transpose(-1, -2).clamp_min(1e-4)
            weight_distribution = Categorical(weights)
            posterior_list = [m.likelihood(m(X)) for m in self.models]
            stacked_means = torch.stack([p.mean for p in posterior_list]).transpose(-1,-2)
            stacked_covar_diags = torch.stack(
                [p.covariance_matrix.diag() for p in posterior_list]
            ).transpose(-1,-2)
            stacked_dist = Normal(stacked_means, stacked_covar_diags)
            return MixtureSameFamily(weight_distribution, stacked_dist)

    def evaluate(self, inputs, targets):
        self.eval()
        inputs = inputs.view(-1, self.stem.input_dim)
        targets = targets.view(-1)
        with torch.no_grad():
            features = self.stem(inputs)
            pred_dist = self(features)
            rmse = (pred_dist.mean - targets).pow(2).mean(0).sqrt().item()
            nll = -pred_dist.log_prob(targets).mean(0).item()
        if not rmse == rmse:
            import pdb; pdb.set_trace()
        return rmse, nll

    def update(self, inputs, targets, *args, **kwargs):
        inputs = inputs.view(-1, self.input_dim)
        targets = targets.view(-1, self.target_dim)
        self._raw_inputs = [torch.cat([*self._raw_inputs, inputs], dim=-2)]
        self._raw_targets = torch.cat([self._raw_targets, targets], dim=-2)
        for i in range(inputs.shape[-2]):
            new_x = self.stem(inputs[i].unsqueeze(0))
            new_y = targets[i].unsqueeze(0)
            _, ranked_models = torch.sort(self._construct_weights(new_x), dim=0, descending=True)
            num_candidates = math.ceil(len(self.models) / 2)

            assignment = None
            for model_idx in ranked_models[:num_candidates]:
                num_data = self.models[model_idx].train_targets.size(-1)
                if num_data >= self.max_data_per_model:
                    continue
                else:
                    assignment = model_idx.squeeze(-1)
                    ######################
                    # dummy to init caches
                    self.models[assignment](new_x)
                    ######################
                    new_model = self.models[assignment].condition_on_observations(new_x, new_y)
                    self.models[assignment] = new_model
                    self.update_model_caches()
                    break

            if assignment is None:
                print("Adding new model")
                assignment = torch.tensor((len(self.models),), device=new_x.device)
                new_model = SingleTaskGP(new_x, new_y, covar_module=self.covar_module)
                new_model.likelihood.initialize(noise=self.noise)
                new_model.eval()
                self.models.append(new_model)
                self.update_model_caches()

            self._assignments = torch.cat([self._assignments, assignment])

        self.train()
        features = self._refresh_features()
        train_dist = self(features)
        loss = -self.mll(train_dist, [m.train_targets for m in self.models])
        loss.backward()
        self.optimizer.step()
        gp_loss = stem_loss = loss.item()

        return gp_loss, stem_loss
    
    def update_model_caches(self):
        self.centers = [m.train_inputs[0].mean(dim=0,keepdim=True) for m in self.models]
        self.num_data = [m.train_inputs[0].shape[-2] for m in self.models]

    def _refresh_features(self):
        features = self.stem(*self._raw_inputs)
        for i, model in enumerate(self.models):
            idx = (self._assignments == i)
            targets = self._reshape_targets(self._raw_targets[idx])
            model.set_train_data(features[idx], targets)
        self.update_model_caches()
        return features
        
    def eval(self, *args, **kwargs):
        self.training = False
        self.stem.eval()
        [m.eval(*args, **kwargs) for m in self.models]
        # return super().eval(*args, **kwargs)
    
    def train(self, *args, **kwargs):
        self.training = True
        self.stem.train()
        [m.train() for m in self.models]

    def fit(self, inputs, targets, num_epochs, test_dataset=None):
        records = []
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, num_epochs, 1e-4)
        for epoch in range(num_epochs):
            self.train()
            self.mll.train()
            self.optimizer.zero_grad()
            features = self._refresh_features()
            train_dist = self(features)
            with gpytorch.settings.skip_logdet_forward(False):
                loss = -self.mll(train_dist, [m.train_targets for m in self.models]).sum()
            loss.backward()
            self.optimizer.step()
            lr_scheduler.step()

            rmse = nll = float('NaN')
            if test_dataset is not None:
                test_x, test_y = test_dataset[:]
                rmse, nll = self.evaluate(test_x, test_y)
            records.append({'train_loss': loss.item(), 'test_rmse': rmse,
                            'test_nll': nll, 'noise': self.models[0].likelihood.noise.mean().item(),
                            'epoch': epoch + 1})

        with torch.no_grad():
            self._refresh_features()

        self.eval()
        return records

    def set_lr(self, gp_lr, stem_lr=None, bn_mom=None):
        stem_lr = gp_lr if stem_lr is None else stem_lr
        self.optimizer = torch.optim.Adam([
            dict(params=self.models.parameters(), lr=gp_lr),
            dict(params=self.stem.parameters(), lr=stem_lr)
        ])
        if bn_mom is not None:
            for m in self.stem.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    m.momentum = bn_mom

    def _reshape_targets(self, targets):
        targets = targets.view(-1, self.target_dim)
        if targets.size(-1) == 1:
            targets = targets.squeeze(-1)
        else:
            targets = targets.t()
        return targets

    def cuda(self):
        return self

    @property
    def noise(self):
        return torch.stack([m.likelihood.noise for m in self.models]).mean(0)

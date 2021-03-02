from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors import GPyTorchPosterior
from botorch import settings

from gpytorch.models import ExactGP
from torch import ones_like

from online_gp.models import FixedNoiseOnlineSKIGP


class OnlineSKIBotorchModel(FixedNoiseOnlineSKIGP):
    def __init__(self,         
        train_inputs=None,
        train_targets=None,
        train_noise_term=None,
        covar_module=None,
        kernel_cache=None,
        grid_bounds=None,
        grid_size=30,
        learn_additional_noise=False,
        **kwargs,
    ):
        super().__init__(
            train_inputs=train_inputs,
            train_targets=train_targets,
            train_noise_term=train_noise_term,
            covar_module=covar_module,
            kernel_cache=kernel_cache,
            grid_bounds=grid_bounds,
            grid_size=grid_size,
            learn_additional_noise=learn_additional_noise,
            **kwargs
        )
        # self.num_outputs = self.train_targets.shape[-1]
        self._is_custom_likelihood = True

    def forward(self, X):
        if X is not None:
            if X.shape[0] == 1:
                X = X[0]

        return super().forward(X)

    def get_fantasy_model(self, inputs, targets, noise=None, **kwargs):
        if noise is None:
            noise = ones_like(targets)
            noise = noise * self.likelihood.noise.mean()

        return super().get_fantasy_model(inputs, targets, noise)

    def fantasize(self, X, sampler, observation_noise=True, **kwargs):
        propagate_grads = kwargs.pop("propagate_grads", False)
        with settings.propagate_grads(propagate_grads):
            post_X = self.posterior(X, observation_noise=observation_noise, **kwargs)
        Y_fantasized = sampler(post_X)  # num_fantasies x batch_shape x n' x m
        # Use the mean of the previous noise values (TODO: be smarter here).
        # noise should be batch_shape x q x m when X is batch_shape x q x d, and
        # Y_fantasized is num_fantasies x batch_shape x q x m.
        noise_shape = Y_fantasized.shape[1:]
        noise = self.likelihood.noise.mean().expand(noise_shape)
        return self.condition_on_observations(X=X, Y=Y_fantasized, noise=noise)

    def posterior(self, X, observation_noise=False, **kwargs):
        self.eval()
        X = X.to(self.likelihood.noise.dtype)
        mvn = self(X)

        return GPyTorchPosterior(mvn)

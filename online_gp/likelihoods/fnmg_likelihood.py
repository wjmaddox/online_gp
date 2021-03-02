import warnings
from typing import Any

import torch
from torch import Tensor

from gpytorch.lazy import ZeroLazyTensor
from gpytorch.utils.warnings import GPInputWarning
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood


class FNMGLikelihood(FixedNoiseGaussianLikelihood):
    """
    Fixed-Noise w/ multiplicative learnable second noise term Gaussian likelihood.
    """
    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise * self.second_noise

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape

        res = self.noise_covar(*params, shape=shape, **kwargs)

        if self.second_noise_covar is not None:
            res = res * self.second_noise_covar(*params, shape=shape, **kwargs)
        elif isinstance(res, ZeroLazyTensor):
            warnings.warn(
                "You have passed data through a FixedNoiseGaussianLikelihood that did not match the size "
                "of the fixed noise, *and* you did not specify noise. This is treated as a no-op.",
                GPInputWarning,
            )

        return res

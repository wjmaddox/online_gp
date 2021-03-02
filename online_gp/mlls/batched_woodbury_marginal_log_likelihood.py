import math
import torch

from gpytorch.mlls import ExactMarginalLogLikelihood

class BatchedWoodburyMarginalLogLikelihood(ExactMarginalLogLikelihood):
    def __init__(self, likelihood, model, clear_caches_every_iteration=False):
        super(BatchedWoodburyMarginalLogLikelihood, self).__init__(likelihood, model)
        self.likelihood = likelihood
        self.model = model

        if self.likelihood.second_noise_covar is not None:
            self.has_learnable_noise = True
        else:
            self.has_learnable_noise = False

        self.clear_caches_every_iteration = clear_caches_every_iteration
        
    def __call__(self, distro, targets, *args):
        if self.clear_caches_every_iteration:    
            self.model.zero_grad() # for the time being to clear caches
        
        current_cache = self.model._kernel_cache

        # I + L'KL
        inner_qmat = self.model.current_qmatrix
        inner_qform, inner_logdet = inner_qmat.inv_quad_logdet(
            inv_quad_rhs=self.model.root_space_projection,
            logdet=True
        )
        inducing_qform = current_cache["interpolation_cache"].transpose(-1, -2).matmul(self.model.Kuu_response)
        inv_quad_term = (current_cache["response_cache"] - inducing_qform).sum((-2, -1)) + inner_qform
        logdet_term = inner_logdet + current_cache["D_logdet"]

        num_data = self.model.num_data

        # add in add'l noise
        final_term = num_data * math.log(2 * math.pi)
        if self.has_learnable_noise:
            inv_quad_term = inv_quad_term / self.likelihood.second_noise_covar.noise

            # should only be an `n` term here when the logdet is calculated b/c 
            # \log |\sigma^{-2} Kuu| = \log Kuu - m \log \sigma^{-2} 
            # which is computed in the `logdet_term` in the forwards
            final_term = num_data * self.likelihood.second_noise_covar.noise.log() + final_term

        res = -0.5 * sum([inv_quad_term, logdet_term, final_term])

        for _, prior, closure, _ in self.named_priors():
            res.add_(prior.log_prob(closure()).sum())

        return res / num_data
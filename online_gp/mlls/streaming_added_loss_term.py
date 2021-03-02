from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls.added_loss_term import AddedLossTerm

from torch.distributions.kl import kl_divergence


class StreamingAddedLossTerm(AddedLossTerm):
    """
    from eqs 13 and 14 of https://arxiv.org/abs/1705.07131
    we simply have to add two new KL loss terms into the standard SVI option
    """

    def __init__(self, variational_dist, old_variational_dist, old_prior, scaling):
        self.old_variational_dist = old_variational_dist
        self.variational_dist = variational_dist
        self.old_prior = old_prior
        self.scaling = scaling

    def loss(self, *params):
        if self.old_variational_dist is not None:
            kl_q_a_new_q_a_old = kl_divergence(
                self.variational_dist, self.old_variational_dist
            )
            kl_q_a_new_p_a_old = kl_divergence(
                self.variational_dist, self.old_prior
            )

            return (
                kl_q_a_new_q_a_old - kl_q_a_new_p_a_old
            ) * self.scaling
        else:
            return 0.0

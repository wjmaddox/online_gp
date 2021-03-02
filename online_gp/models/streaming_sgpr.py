from gpytorch.models import ApproximateGP
from gpytorch import variational, means, kernels, likelihoods, distributions, lazy
import torch
from copy import deepcopy
from gpytorch.utils.cholesky import psd_safe_cholesky


class StreamingSGPR(ApproximateGP):
    """
    https://github.com/thangbui/streaming_sparse_gp/blob/b46e6e4a9257937f7ca26ac06099f5365c8b50d8/code/osgpr.py
    """
    def __init__(
            self,
            inducing_points,
            old_strat=None,
            old_kernel=None,
            old_C_matrix=None,
            covar_module=None,
            likelihood=None,
            learn_inducing_locations=True,
            num_data=0,
            jitter=1e-4
    ):
        data_dim = -2 if inducing_points.dim() > 1 else -1
        variational_distribution = variational.CholeskyVariationalDistribution(
            inducing_points.size(data_dim)
        )
        variational_strategy = variational.UnwhitenedVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        self.mean_module = means.ZeroMean()

        if likelihood is None:
            self.likelihood = likelihoods.GaussianLikelihood()
        else:
            self.likelihood = likelihood

        if covar_module is None:
            self.covar_module = kernels.MaternKernel()
        else:
            self.covar_module = covar_module

        self._old_strat = old_strat
        self._old_kernel = old_kernel
        self._old_C_matrix = old_C_matrix
        self._data_dim = data_dim
        self._jitter = jitter
        self.num_data = num_data

    def forward(self, inputs):
        mean = self.mean_module(inputs)
        covar = self.covar_module(inputs)
        return distributions.MultivariateNormal(mean, covar)

    def current_C_matrix(self, x):
        sigma2 = self.likelihood.noise
        z_b = self.variational_strategy.inducing_points
        Kbf = self.covar_module(z_b, x).evaluate()
        C1 = Kbf @ Kbf.transpose(-1, -2) / sigma2

        if self._old_C_matrix is None:
            C2 = torch.zeros_like(C1)
        else:
            assert self._old_strat is not None
            assert self._old_kernel is not None
            z_a = self._old_strat.inducing_points.detach()
            Kaa_old = self._old_kernel(z_a).add_jitter(self._jitter).detach()
            C_old = self._old_C_matrix.detach()
            Kab = self.covar_module(z_a, z_b).evaluate()
            Kaa_old_inv_Kab = Kaa_old.inv_matmul(Kab)
            C2 = Kaa_old_inv_Kab.transpose(-1, -2) @ C_old.matmul(Kaa_old_inv_Kab)

        C = C1 + C2
        L = psd_safe_cholesky(C, upper=False, jitter=self._jitter)
        L = lazy.TriangularLazyTensor(L, upper=False)
        return lazy.CholLazyTensor(L, upper=False)

    def current_c_vec(self, x, y):
        sigma2 = self.likelihood.noise
        z_b = self.variational_strategy.inducing_points
        Kbf = self.covar_module(z_b, x).evaluate()
        c1 = Kbf @ y / sigma2

        if self._old_C_matrix is None:
            c2 = torch.zeros_like(c1)
            c3 = torch.zeros_like(c1)
        else:
            assert self._old_strat is not None
            assert self._old_kernel is not None
            z_a = self._old_strat.inducing_points.detach()
            ma = self._old_strat.variational_distribution.mean.detach().unsqueeze(-1)
            Kaa_old = self._old_kernel(z_a).add_jitter(self._jitter).detach()
            C_old = self._old_C_matrix.detach()
            Kab = self.covar_module(z_a, z_b).evaluate()
            Kaa_old_inv_ma = Kaa_old.inv_matmul(ma)
            Kba_Kaa_old_inv = Kaa_old.inv_matmul(Kab).transpose(-1, -2)

            c2 = Kab.transpose(-1, -2) @ Kaa_old_inv_ma
            c3 = Kba_Kaa_old_inv @ C_old.matmul(Kaa_old_inv_ma)

        return c1 + c2 + c3

    @property
    def pseudotargets(self):
        if self._old_strat is None:
            return None

        ma = self._old_strat.variational_distribution.mean.detach().unsqueeze(-1)
        z_a = self._old_strat.inducing_points.detach()
        Kaa_old = self._old_kernel(z_a).evaluate().detach()
        C_old = self._old_C_matrix.detach()
        C_old_inv_ma = C_old.inv_matmul(ma)

        return Kaa_old @ C_old_inv_ma + ma

    def _update_variational_moments(self, x, y):
        C = self.current_C_matrix(x)
        c = self.current_c_vec(x, y)
        z_b = self.variational_strategy.inducing_points
        Kbb = self.covar_module(z_b).evaluate()
        L = psd_safe_cholesky(Kbb + C.evaluate(), upper=False, jitter=self._jitter)
        m_b = Kbb @ torch.cholesky_solve(c, L, upper=False)
        S_b = Kbb @ torch.cholesky_solve(Kbb, L, upper=False)

        return m_b, S_b

    def update_variational_distribution(self, x_new, y_new):
        m_b, S_b = self._update_variational_moments(x_new, y_new)

        q_mean = self.variational_strategy._variational_distribution.variational_mean
        q_mean.data.copy_(m_b.squeeze(-1))

        upper_new_covar = psd_safe_cholesky(S_b, jitter=self._jitter)
        upper_q_covar = self.variational_strategy._variational_distribution.chol_variational_covar
        upper_q_covar.copy_(upper_new_covar)
        self.variational_strategy.variational_params_initialized.fill_(1)

    def get_fantasy_model(self, x_new, y_new, resample_ratio=0.2):
        assert resample_ratio <= 1.
        z_old = self.variational_strategy.inducing_points.clone().detach()
        perturbation = torch.empty_like(z_old)
        perturbation.uniform_(-1e-4, 1e-4)
        z_old += perturbation

        num_resampled = min(int(resample_ratio * z_old.size(0)), x_new.size(0))
        z_resample_idxs = torch.randperm(z_old.size(0))[:num_resampled]
        z_mask = torch.ones(z_old.size(0), dtype=bool)
        z_mask[z_resample_idxs] = 0
        x_resample_idxs = torch.randperm(x_new.size(0))[:num_resampled]

        z_new = torch.empty_like(z_old)
        z_new[z_mask] = z_old[z_mask]
        z_new[~z_mask] = x_new[x_resample_idxs]

        # add more and more inducing points
        # z_new = z_old.clone().detach()
        # if torch.rand(1) < 0.5:
        #     z_new = torch.cat([z_new, x_new])

        fantasy_model = StreamingSGPR(
            inducing_points=z_new,
            likelihood=deepcopy(self.likelihood),
            covar_module=deepcopy(self.covar_module),
            old_strat=self.variational_strategy,
            old_kernel=self.covar_module,
            old_C_matrix=self.current_C_matrix(x_new),
            num_data=self.num_data + x_new.size(0),
            jitter=self._jitter
        )
        with torch.no_grad():
            fantasy_model.update_variational_distribution(x_new, y_new)
        return fantasy_model

    def predict(self, inputs):
        self.eval()
        pred_dist = self.likelihood(self(inputs))
        return pred_dist.mean, pred_dist.variance

    def disable_q_grad(self):
        self.variational_strategy._variational_distribution.variational_mean.requires_grad_(False)
        self.variational_strategy._variational_distribution.chol_variational_covar.requires_grad_(False)


class StreamingSGPRBound(object):
    def __init__(self, ssgp, combine_terms=True):
        self.gp = ssgp
        self._combine_terms = combine_terms

    def __call__(self, x, y):
        sigma2 = self.gp.likelihood.noise
        z_b = self.gp.variational_strategy.inducing_points
        Kff = self.gp.covar_module(x).evaluate()
        Kbf = self.gp.covar_module(z_b, x).evaluate()
        Kbb = self.gp.covar_module(z_b).add_jitter(self.gp._jitter)

        Q1 = Kbf.transpose(-1, -2) @ Kbb.inv_matmul(Kbf)
        Sigma1 = sigma2 * torch.eye(Q1.size(-1)).to(Q1.device)

        # logp term
        if self.gp._old_strat is None:
            num_data = y.size(-2)
            mean = torch.zeros(num_data).to(y.device)
            covar = (Q1 + Sigma1) + self.gp._jitter * torch.eye(Q1.size(-2)).to(Q1.device)
            dist = distributions.MultivariateNormal(mean, covar)
            logp_term = dist.log_prob(y.squeeze(-1)).sum() / y.size(-2)
        else:
            z_a = self.gp._old_strat.inducing_points.detach()
            Kba = self.gp.covar_module(z_b, z_a).evaluate()
            Kaa_old = self.gp._old_kernel(z_a).evaluate().detach()
            Q2 = Kba.transpose(-1, -2) @ Kbb.inv_matmul(Kba)
            zero_1 = torch.zeros(Q1.size(-2), Q2.size(-1)).to(Q1.device)
            zero_2 = torch.zeros(Q2.size(-2), Q1.size(-1)).to(Q1.device)
            Q = torch.cat(
                [
                    torch.cat([Q1, zero_1], dim=-1),
                    torch.cat([zero_2, Q2], dim=-1)
                ], dim=-2
            )

            C_old = self.gp._old_C_matrix.detach()
            Sigma2 = Kaa_old @ C_old.inv_matmul(Kaa_old)
            Sigma2 = Sigma2 + self.gp._jitter * torch.eye(Sigma2.size(-2)).to(Sigma2.device)
            Sigma = torch.cat(
                [
                    torch.cat([Sigma1, zero_1], dim=-1),
                    torch.cat([zero_2, Sigma2], dim=-1)
                ], dim=-2
            )

            y_hat = torch.cat([y, self.gp.pseudotargets])
            mean = torch.zeros_like(y_hat.squeeze(-1))
            covar = (Q + Sigma) + self.gp._jitter * torch.eye(Q.size(-2)).to(Q.device)
            dist = distributions.MultivariateNormal(mean, covar)
            logp_term = dist.log_prob(y_hat.squeeze(-1)).sum() / y_hat.size(-2)
            num_data = y_hat.size(-2)

        # trace term
        t1 = (Kff - Q1).diag().sum() / sigma2
        t2 = 0
        if self.gp._old_strat is not None:
            LSigma2 = psd_safe_cholesky(Sigma2, upper=False, jitter=self.gp._jitter)
            Kaa = self.gp.covar_module(z_a).evaluate().detach()
            Sigma2_inv_Kaa = torch.cholesky_solve(Kaa, LSigma2, upper=False)
            Sigma2_inv_Q2 = torch.cholesky_solve(Q2, LSigma2, upper=False)
            t2 = Sigma2_inv_Kaa.diag().sum() - Sigma2_inv_Q2.diag().sum()
        trace_term = -(t1 + t2) / 2 / num_data

        if self._combine_terms:
            return logp_term + trace_term
        else:
            return logp_term, trace_term, t1 / num_data, t2 / num_data

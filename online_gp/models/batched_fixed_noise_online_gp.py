import torch

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, GridInterpolationKernel
from gpytorch.means import ZeroMean
from gpytorch.lazy import DiagLazyTensor, lazify, RootLazyTensor
from online_gp.likelihoods import FNMGLikelihood
from gpytorch.models import GP
from gpytorch.utils.memoize import (
    cached,
    pop_from_cache,
)
from gpytorch.utils.errors import CachingError
from gpytorch.utils.interpolation import left_interp
from gpytorch.settings import skip_posterior_variances, fast_pred_var, fast_pred_samples
from gpytorch.lazy import ZeroLazyTensor

from online_gp.lazy import UpdatedRootLazyTensor
from online_gp.settings import detach_interp_coeff


def _get_wmat_from_kernel(lazy_kernel):
    wmat =  lazy_kernel._sparse_left_interp_t(
        lazy_kernel.left_interp_indices, lazy_kernel.left_interp_values
    ).to_dense()
    if detach_interp_coeff.on():
        wmat = wmat.detach()
    return wmat


def _initialize_caches(targets, noise_diagonal, wmat, create_w_cache=True):
    if len(noise_diagonal.shape) > 2:
        noise_diagonal = noise_diagonal.squeeze(-1)
    noise_diagonal = DiagLazyTensor(noise_diagonal)

    # reshape the targets so that we have niceness in the batch dimensions
    if targets.ndimension() == 2:
        targets = targets.unsqueeze(-1)
    if targets.ndimension() <= 3:
        targets = targets.transpose(-2, -3)

    dinv_y = noise_diagonal.inv_matmul(targets)

    cache_dict = {
        "response_cache": targets.transpose(-1, -2) @ dinv_y,
        "interpolation_cache": wmat @ dinv_y,
    }

    if create_w_cache:
        cache_dict["WtW"] = UpdatedRootLazyTensor(
            wmat @ (noise_diagonal.inv_matmul(wmat.transpose(-1, -2))),
            initial_is_root=False,
        )

    cache_dict["D_logdet"] = noise_diagonal.logdet()

    # trim tails in the case of large batch dim
    if targets.ndimension() > 3:
        cache_dict["response_cache"] = cache_dict["response_cache"].squeeze(-1)
    return cache_dict


class FixedNoiseOnlineSKIGP(GP):
    def __init__(
        self,
        train_inputs=None,
        train_targets=None,
        train_noise_term=None,
        covar_module=None,
        kernel_cache=None,
        grid_bounds=None,
        grid_size=30,
        likelihood=None,
        learn_additional_noise=False,
        num_data = None,
    ):
        super().__init__()

        assert train_inputs is not None or kernel_cache is not None

        if train_targets is not None:
            num_outputs = train_targets.shape[-1]
            input_batch_shape = train_inputs.shape[:-2]
            self.num_data = train_inputs.shape[-2]

        else:
            # pull from kernel_cache
            num_outputs = kernel_cache["response_cache"].shape[-1]
            input_batch_shape = kernel_cache["WtW"].shape[0]
            self.num_data = num_data

        self.num_outputs = num_outputs

        _batch_shape = input_batch_shape
        if num_outputs > 1:
            _batch_shape += torch.Size([num_outputs])

        if covar_module is None:
            if grid_bounds is None:
                grid_bounds = torch.stack(
                    (
                        train_inputs.min(dim=-2)[0] - 0.1,
                        train_inputs.max(dim=-2)[0] + 0.1,
                    )
                ).transpose(-1,-2)

            covar_module = ScaleKernel(
                RBFKernel(
                    batch_shape=_batch_shape, ard_num_dims=train_inputs.size(-1)
                ),
                batch_shape=_batch_shape,
            )

        if type(covar_module) is not GridInterpolationKernel:
            covar_module = GridInterpolationKernel(
                base_kernel=covar_module,
                grid_size=grid_size,
                num_dims=train_inputs.shape[-1],
                grid_bounds=grid_bounds,
            )

        self._batch_shape = _batch_shape
        self.train_inputs = [None]
        self.train_targets = None

        self.covar_module = covar_module
        self.mean_module = ZeroMean()
        if likelihood is None:
            if train_noise_term is None:
                train_noise_term = torch.ones_like(train_targets)

            self.likelihood = FNMGLikelihood(
                noise=train_noise_term.transpose(-1, -2),
                learn_additional_noise=learn_additional_noise,
            )
        else:
            self.likelihood = likelihood
        self.has_learnable_noise = learn_additional_noise

        # initialize the kernel caches immediately so we can throw away the data
        if kernel_cache is None:
            self.covar_module = self.covar_module.to(train_inputs.device)
            initial_kxx = self.covar_module(train_inputs).evaluate_kernel()
            initial_wmat = _get_wmat_from_kernel(initial_kxx)
            self._kernel_cache = _initialize_caches(
                train_targets,
                train_noise_term.transpose(-1, -2),
                initial_wmat,
                create_w_cache=True
            )      
        else:
            self._kernel_cache = kernel_cache

    # TODO: make _cache_dict a cached object
    def _update_cache_dicts(self, targets, noise_diagonal, new_wmat):
        new_cache_dict = _initialize_caches(targets, noise_diagonal, new_wmat, create_w_cache=False)
        updated_cache_dict = {}
        for key in self._kernel_cache.keys():
            if key != "WtW":
                updated_cache_dict[key] = new_cache_dict[key] + self._kernel_cache[key]
            else:
                # we need to update "WtW" separately
                root_noise_diagonal = noise_diagonal.clamp_min(1e-7) ** 0.5
                if len(root_noise_diagonal.shape) < 3:
                    root_noise_diagonal = root_noise_diagonal.unsqueeze(-2)
                else:
                    root_noise_diagonal = root_noise_diagonal.transpose(-1, -2)
                new_w_dinv = new_wmat / root_noise_diagonal
                updated_cache_dict[key] = self._kernel_cache[key].update(new_w_dinv)

        return updated_cache_dict

    def forward(self, X, **kwargs):
        if self.training:
            # TODO: return a better dummy here
            # is a dummy b/c the real action happens in the MLL
            if X is not None:
                mean = self.mean_module(X)
                covar = self.covar_module(X)
            else:
                if type(self._batch_shape) is not torch.Size:
                    batch_shape = torch.Size((self._batch_shape,))
                else:
                    batch_shape = self._batch_shape
                mean_shape = batch_shape + torch.Size((self.num_data,))
                mean = ZeroLazyTensor(*mean_shape)
                covar_shape = mean_shape + torch.Size((self.num_data,))
                covar = ZeroLazyTensor(*covar_shape)

            # should hopefuly only occur in batching issues
            if (
                mean.ndimension() < covar.ndimension()
                and (
                    self._batch_shape != torch.Size()
                    and mean.shape != covar.shape[:-1]
                )                
            ):
                if type(mean) is ZeroLazyTensor:
                    mean = mean.evaluate()
                mean = mean.unsqueeze(0)
                mean = mean.repeat(covar.batch_shape[0], *[1]*(covar.ndimension() - 1))

            return MultivariateNormal(mean, covar)
        else:
            lazy_kernel = self.covar_module(X).evaluate_kernel()
            pred_mean = left_interp(
                lazy_kernel.left_interp_indices,
                lazy_kernel.left_interp_values,
                self.prediction_cache["pred_mean"],
            )

            if skip_posterior_variances.off():
                # init predictive covariance if it's not in the prediction cache
                if "pred_cov" in self.prediction_cache.keys():
                    inner_pred_cov = self.prediction_cache["pred_cov"]
                else:
                    self.prediction_cache[
                        "pred_cov"
                    ] = self._make_predictive_covar()
                    inner_pred_cov = self.prediction_cache["pred_cov"]

                if fast_pred_samples.off():
                    pred_wmat = _get_wmat_from_kernel(lazy_kernel)
                    lazy_pred_wmat = lazify(pred_wmat)
                    pred_cov = lazy_pred_wmat.transpose(-1, -2).matmul((inner_pred_cov.matmul(lazy_pred_wmat)))

                    if self.has_learnable_noise:
                        pred_cov = pred_cov * self.likelihood.second_noise_covar.noise.to(pred_cov.device)
                else:
                    # inner_pred_cov_root = inner_pred_cov.root_decomposition().root.evaluate()
                    inner_pred_cov_root = inner_pred_cov.root_decomposition(method="lanczos").root.evaluate()
                    if inner_pred_cov_root.shape[-1] > X.shape[-2]:
                        inner_pred_cov_root = inner_pred_cov_root[..., -X.shape[-2]:]

                    root_tensor = left_interp(
                        lazy_kernel.left_interp_indices,
                        lazy_kernel.left_interp_values,
                        inner_pred_cov_root,
                    )
                    
                    if self.has_learnable_noise:
                        noise_root = self.likelihood.second_noise_covar.noise.to(root_tensor.device) ** 0.5
                    pred_cov = RootLazyTensor(root_tensor * noise_root)

            else:
                pred_cov = ZeroLazyTensor(*lazy_kernel.size())

            pred_mean = pred_mean[..., 0]
            if self._batch_shape == torch.Size() and X.ndimension() == 2:
                pred_mean = pred_mean[0]
                if pred_cov.ndimension() > 2:
                    pred_cov = pred_cov[0]

            dist = MultivariateNormal(pred_mean, pred_cov)

            return dist

    def condition_on_observations(self, X, Y, noise=None, inplace=False):
        if noise is None:
            noise = torch.ones_like(Y)
        lazy_kernel = self.covar_module(X).evaluate_kernel()
        if (noise.shape[:-2] != self._batch_shape or noise.shape[-1]==1) and noise.ndimension() < 3:
            noise_for_update = noise.transpose(-1, -2)
        else:
            noise_for_update = noise
        new_kernel_cache = self._update_cache_dicts(
            Y, noise_for_update, _get_wmat_from_kernel(lazy_kernel)
        )

        if inplace:
            self.num_data = self.num_data+X.shape[-2]
            self._kernel_cache = new_kernel_cache
            self._dump_caches()

        else:
            new_gp = type(self)(
                covar_module=self.covar_module,
                kernel_cache=new_kernel_cache,
                learn_additional_noise=self.has_learnable_noise,
                likelihood=self.likelihood,
                num_data=self.num_data+X.shape[-2],
            )
            if self.has_learnable_noise:
                new_gp.likelihood.second_noise_covar.noise = self.likelihood.second_noise_covar.noise.to(X.device)
            return new_gp

    def get_fantasy_model(self, inputs, targets, noise_term, **kwargs):
        target_batch_shape = targets.shape[:-1]
        input_batch_shape = inputs.shape[:-2]
        tbdim, ibdim = len(target_batch_shape), len(input_batch_shape)

        if not (tbdim == ibdim + 1 or tbdim == ibdim):
            raise RuntimeError(
                f"Unsupported batch shapes: The target batch shape ({target_batch_shape}) must have either the "
                f"same dimension as or one more dimension than the input batch shape ({input_batch_shape})"
            )

        expanded_kernel_cache = {}
        expanded_kernel_cache["response_cache"] = self._kernel_cache[
            "response_cache"
        ].expand(target_batch_shape + self.num_outputs)
        new_wtw_cache = self._kernel_cache["WtW"].expand(
            input_batch_shape + self._kernel_cache["WtW"].shape[-2:]
        )
        expanded_kernel_cache["WtW"] = new_wtw_cache
        new_interpolation_cache = self._kernel_cache[
            "interpolation_cache"
        ].expand(
            target_batch_shape[:-1]
            + new_wtw_cache.shape[:-1]
            + self.num_outputs
        )
        expanded_kernel_cache["interpolation_cache"] = new_interpolation_cache
        expanded_kernel_cache["D_logdet"] = self._kernel_cache[
            "D_logdet"
        ].expand(target_batch_shape)

        # we now create a new model and update the caches
        expanded_model = type(self)(
            covar_module=self.covar_module,
            kernel_cache=expanded_kernel_cache,
            learn_additional_noise=self.has_learnable_noise,
            likelihood=self.likelihood,
            num_data=self.num_data
        )
        if self.has_learnable_noise:
            expanded_model.likelihood.second_noise_covar.noise = self.likelihood.second_noise_covar.noise.to(Kuu.device)

        expanded_model.condition_on_observations(
            inputs, targets.unsqueeze(-1), noise_term, inplace=True
        )
        return expanded_model

    @property
    @cached(name="Kuu")
    def Kuu(self):
        Kuu = self.covar_module._inducing_forward(last_dim_is_batch=False)
        if self.has_learnable_noise:
            # append 1 / \sigma^2 into the Kuu term in the qmatrix
            Kuu = Kuu / self.likelihood.second_noise_covar.noise
        return Kuu

    @property
    @cached(name="current_inducing_compression_matrix")
    def current_inducing_compression_matrix(self):
        wtw = self._kernel_cache["WtW"]
        Lmat = wtw.root_decomposition().root
        return self.Kuu @ Lmat

    @property
    @cached(name="current_qmatrix")
    def current_qmatrix(self):
        Kuu_L = self.current_inducing_compression_matrix
        Lmat = self._kernel_cache["WtW"].root_decomposition().root
        return (Lmat.transpose(-1, -2) @ Kuu_L).add_jitter(1.0)

    @property
    @cached(name="root_space_projection")
    def root_space_projection(self):
        Lmat = self._kernel_cache["WtW"].root_decomposition().root
        return Lmat.transpose(-1, -2) @ self.Kuu_response

    @property
    @cached(name="Kuu_response")
    def Kuu_response(self):
        return self.Kuu.matmul(self._kernel_cache["interpolation_cache"])

    @property
    @cached(name="prediction_cache")
    def prediction_cache(self):
        prediction_cache = {}

        Kuu_Lmat = self.current_inducing_compression_matrix.evaluate()

        qmat_solve = self.current_qmatrix.inv_matmul(self.root_space_projection)
        predictive_mean_cache = self.Kuu_response - Kuu_Lmat @ qmat_solve
        prediction_cache["pred_mean"] = predictive_mean_cache

        if skip_posterior_variances.off():
            prediction_cache["pred_cov"] = self._make_predictive_covar(
                self.current_qmatrix, self.Kuu, Kuu_Lmat
            )
        return prediction_cache

    def _make_predictive_covar(self, qmatrix=None, Kuu=None, Kuu_Lmat=None):
        if qmatrix is None:
            qmatrix = self.current_qmatrix
        if Kuu is None:
            Kuu = self.Kuu
        if Kuu_Lmat is None:
            Kuu_Lmat = self.current_inducing_compression_matrix.evaluate()

        if fast_pred_var.on():
            qmat_inv_root = qmatrix.root_inv_decomposition()
            # to lazify you have to evaluate the inverse root which is slow
            # otherwise, you can't backprop your way through it
            inner_cache = RootLazyTensor(Kuu_Lmat.matmul(qmat_inv_root.root.evaluate()))
        else:
            inner_cache = Kuu_Lmat.matmul(
                qmatrix.inv_matmul(Kuu_Lmat.transpose(-1, -2))
            )

        predictive_covar_cache = Kuu - inner_cache
        return predictive_covar_cache

    def _dump_caches(self):
        # this is done in gpytorch.variational.variational_strategy as well
        fixed_cache_names = ["current_qmatrix", "current_inducing_compression_matrix", "prediction_cache",
                             "root_space_projection", "Kuu_response", "Kuu"]
        for name in fixed_cache_names:
            try:
                pop_from_cache(self, name)
            except CachingError:
                pass

    def zero_grad(self):
        self._dump_caches()
        return super().zero_grad()

    def set_train_data(self, train_inputs, train_targets, train_noise_term):
        initial_kxx = self.covar_module(train_inputs).evaluate_kernel()
        initial_wmat = _get_wmat_from_kernel(initial_kxx)
        self._kernel_cache = _initialize_caches(
            train_targets,
            train_noise_term.transpose(-1, -2),
            initial_wmat,
            create_w_cache=True
        )
        
    def to(self, device):
        if self._kernel_cache is not None:
            for key in self._kernel_cache:
                self._kernel_cache[key] = self._kernel_cache[key].to(device)
                
        return super().to(device)

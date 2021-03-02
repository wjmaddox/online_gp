import torch
from online_gp.models.batched_fixed_noise_online_gp import _get_wmat_from_kernel
from gpytorch.settings import skip_posterior_variances


def sm_partial_mll(ski_gp, new_x, new_y, num_seen):
    # M := (K_{uu}^{-1} + W'W)^{-1} = K_{uu} - K_{uu}LQ^{-1}L'K_{uu}
    with skip_posterior_variances(False):
        M = ski_gp.prediction_cache['pred_cov'].detach()
    W_y = ski_gp._kernel_cache["interpolation_cache"].detach()

    # Q = ski_gp.current_qmatrix.detach()
    # Kuu_L = ski_gp.current_inducing_compression_matrix.detach()
    # Kuu_L_t = Kuu_L.transpose(-1, -2)
    # Kuu = ski_gp.kxx_cache.base_lazy_tensor.detach()
    # if ski_gp.has_learnable_noise:
    #     Kuu = Kuu / ski_gp.likelihood.second_noise_covar.noise.detach()

    # w:= w(x')
    lazy_kernel = ski_gp.covar_module(new_x).evaluate_kernel()
    w = _get_wmat_from_kernel(lazy_kernel)
    if w.ndim < 3:
        w = w.unsqueeze(0)

    new_W_y = W_y + w * new_y
    new_W_y_t = new_W_y.transpose(-1, -2)

    rhs = torch.cat([w, new_W_y], dim=-1)
    solves = M.matmul(rhs)

    # v := Mw
    v = solves[..., :1]
    # v_rhs = Kuu_L_t.matmul(w)
    # v = Kuu.matmul(w) - Kuu_L.matmul(Q.inv_matmul(v_rhs))
    v_t = v.transpose(-1, -2)
    sm_divisor = 1 + v_t.bmm(w)

    # quad_term_1 := y'WK_{uu}W'y
    # quad_term_1 = new_W_y_t.matmul(Kuu.matmul(new_W_y))
    # # quad_term_2 := y'WK_{uu}LQ^{-1}L'K_{uu}W'y
    # term_2_rhs = Kuu_L_t.matmul(new_W_y)
    # term_2_rhs_t = term_2_rhs.transpose(-1, -2)
    # quad_term_2 = term_2_rhs_t.matmul(Q.inv_matmul(term_2_rhs))
    # quad_term_3 := y'Wvv'W'y / (1 + v'w)

    M_W_y = solves[..., 1:]
    quad_term_1 = new_W_y_t.matmul(M_W_y)

    quad_term_3 = (v_t.bmm(new_W_y) ** 2) / sm_divisor

    # quad_term := y'WAW'y - (y'Wvv'W'y) / (1 + v'w)
    # quad_term = (quad_term_1 - quad_term_2 - quad_term_3)
    quad_term = quad_term_1 - quad_term_3
    if ski_gp.has_learnable_noise:
        quad_term = quad_term / ski_gp.likelihood.second_noise_covar.noise.detach()

    # \log|WKW' + \sigma^2 I| = n\log(\sigma^2) + \log|K_{uu}| - \log|A_t|
    # \log|A_t| = \log|A_{t-1}| - \log(1 + v'w)
    logdet_term = torch.log(sm_divisor)

    partial_mll = (quad_term - logdet_term) / 2
    return partial_mll / (num_seen + 1)

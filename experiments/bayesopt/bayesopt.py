import time
import argparse
import torch

from botorch import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch
from botorch.models import SingleTaskGP, FixedNoiseGP
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.priors import GammaPrior
from gpytorch.settings import (
    max_cholesky_size,
    detach_test_caches,
    skip_logdet_forward,
    use_toeplitz,
    max_root_decomposition_size,
    cholesky_jitter,
    fast_pred_var,
    fast_pred_samples,
)
from gpytorch.utils.grid import create_grid
from pandas import DataFrame

from online_gp.models import OnlineSKIBotorchModel
from online_gp.mlls import BatchedWoodburyMarginalLogLikelihood
from online_gp.models.variational_gp_model import (
    VariationalGPModel,
    ApproximateGPyTorchModel,
)

from utils import (
    prepare_function,
    prepare_acquisition_function,
    initialize_random_data,
    parse,
    optimize_acqf_and_get_observation,
)


def main(args):
    if args.batch_size > 1 and args.acqf == "mves":
        raise NotImplementedError(
            "Cyclic optimization is not implemented for MVES currently. Please use a batch size of 1."
        )
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    torch.random.manual_seed(args.seed)
    test_function = prepare_function(args, args.device)
    init_x, init_y, y_means, latent_y = initialize_random_data(
        test_function, args.device, args.num_init
    )

    bounds = test_function.bounds.t()

    unit_bounds = torch.ones_like(bounds)
    unit_bounds[:, 0] = 0.0

    noise = args.noise ** 2 * torch.ones_like(init_y) if args.fixed_noise else None

    if args.model == "wiski":

        def initialize_model(X, Y, old_model=None, **kwargs):
            if old_model is None:
                covar_module = ScaleKernel(
                    MaternKernel(
                        nu=2.5,
                        lengthscale_prior=GammaPrior(3.0, 6.0),
                        lengthscale_constraint=Interval(1e-4, 12.0),
                    ),
                    outputscale_prior=GammaPrior(2.0, 0.15),
                    outputscale_constraint=Interval(1e-4, 12.0),
                )
            else:
                covar_module = old_model.covar_module

            if args.dim == 3:
                wiski_grid_size = 10
            elif args.dim == 2:
                wiski_grid_size = 30

            kernel_cache = old_model._kernel_cache if old_model is not None else None

            model_obj = OnlineSKIBotorchModel(
                X,
                Y,
                train_noise_term=noise,
                grid_bounds=bounds,
                grid_size=wiski_grid_size,
                learn_additional_noise=True,
                kernel_cache=kernel_cache,
                covar_module=covar_module,
            ).to(X)

            mll = BatchedWoodburyMarginalLogLikelihood(
                model_obj.likelihood, model_obj, clear_caches_every_iteration=True
            )
            # TODO: reload statedict here?
            # weird errors resulting

            return model_obj, mll

    elif args.model == "exact":

        def initialize_model(X, Y, old_model=None, **kwargs):
            if old_model is None:
                covar_module = ScaleKernel(
                    MaternKernel(
                        nu=2.5,
                        lengthscale_prior=GammaPrior(3.0, 6.0),
                        lengthscale_constraint=Interval(1e-4, 12.0),
                    ),
                    outputscale_prior=GammaPrior(2.0, 0.15),
                    outputscale_constraint=Interval(1e-4, 12.0),
                )

                if args.fixed_noise:
                    model_obj = FixedNoiseGP(
                        X, Y, train_Yvar=noise, covar_module=covar_module
                    )
                else:
                    model_obj = SingleTaskGP(X, Y, covar_module=covar_module)
            else:
                model_obj = old_model
            mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
            return model_obj, mll

    elif args.model == "osvgp":

        def initialize_model(X, Y, old_model=None, **kwargs):
            if old_model is None:
                if args.dim == 3:
                    wiski_grid_size = 10
                elif args.dim == 2:
                    wiski_grid_size = 30

                grid_list = create_grid([wiski_grid_size] * args.dim, grid_bounds=bounds)
                inducing_points = (
                    torch.stack([x.reshape(-1) for x in torch.meshgrid(grid_list)])
                    .t()
                    .contiguous()
                    .clone()
                )

                likelihood = GaussianLikelihood()
                model_base = VariationalGPModel(
                    inducing_points,
                    likelihood=likelihood,
                    beta=1.0,
                    learn_inducing_locations=True,
                )
                model_obj = ApproximateGPyTorchModel(
                    model_base, likelihood, num_outputs=1
                )
                model_base.train_inputs = [X]
                model_base.train_targets = Y.view(-1)

                # we don't implement fixednoiseGaussian likelihoods for the streaming setting
                if args.fixed_noise:
                    model_obj.likelihood.noise = args.noise ** 2
                    model_obj.likelihood.requires_grad = False
            else:
                model_obj = old_model
                model_obj.train_inputs = [X]
                model_obj.train_targets = Y.view(-1)

            mll = VariationalELBO(
                model_obj.likelihood, model_obj.model, num_data=X.shape[-2]
            )
            return model_obj, mll

    train_x, train_y = init_x, init_y
    model_obj = None

    all_outputs = []
    for step in range(args.num_steps):
        t0 = time.time()
        model_obj, mll = initialize_model(train_x, train_y, old_model=model_obj)
        model_obj = model_obj.to(train_x)

        # fitting with LBFGSB is really slow due to the inducing points
        if args.model != "osvgp":
            fit_gpytorch_model(mll)
        else:
            fit_gpytorch_torch(mll, options={"maxiter": 1000})

        t0_total = time.time() - t0

        acqf = prepare_acquisition_function(
            args, model_obj, train_x, train_y, bounds, step
        )

        t1 = time.time()

        (
            new_x_ei,
            new_obj_unstandardized,
            new_latent_obj,
        ) = optimize_acqf_and_get_observation(
            acqf,
            bounds=unit_bounds.t(),
            test_function_bounds=bounds.t(),
            batch_size=args.batch_size,
            test_function=test_function,
        )
        new_obj_ei = (new_obj_unstandardized - y_means["mean"]) / y_means["std"]

        train_x = torch.cat((train_x, new_x_ei), dim=0)
        train_y = torch.cat((train_y, new_obj_ei), dim=0)
        latent_y = torch.cat((latent_y, new_latent_obj), dim=0)
        if noise is not None:
            new_noise = args.noise ** 2 * torch.ones_like(new_obj_ei)
            noise = torch.cat((noise, new_noise), dim=0)
        else:
            new_noise = None
        t1_total = time.time() - t1

        t2 = time.time()
        if args.model != "osvgp":
            if args.fixed_noise:
                kwargs = {"noise": new_noise}
            else:
                kwargs = {}
            model_obj = model_obj.condition_on_observations(
                X=new_x_ei, Y=new_obj_ei, **kwargs
            )
        if args.model == "osvgp":
            model_obj.model.update_variational_parameters(
                new_x=new_x_ei, new_y=new_obj_ei
            )
        t2_total = time.time() - t2
        total = t0_total + t1_total + t2_total

        max_achieved = train_y.max() * y_means["std"] + y_means["mean"]
        max_latent_achieved = latent_y.max()
        output_lists = [
            t0_total,
            t1_total,
            t2_total,
            total,
            max_achieved.item(),
            max_latent_achieved.item(),
        ]
        all_outputs.append(output_lists)

        if step % (args.num_steps // 5) == 0:
            print(
                "Step ",
                step,
                " of ",
                args.num_steps,
                "Max Achieved: ",
                max_achieved.item(),
                "Max Latent Achieved: ",
                max_latent_achieved.item(),
            )

    for key in y_means:
        y_means[key] = y_means[key].cpu()

    output_dict = {
        "observations": {
            "x": train_x.cpu(),
            "y": train_y.cpu(),
            "means": y_means,
            "latent_y": latent_y.cpu(),
        },
        "results": DataFrame(all_outputs),
        "args": args
    }
    torch.save(output_dict, args.output)


if __name__ == "__main__":
    args = parse()
    use_fast_pred_var = True if not args.use_exact else False

    with use_toeplitz(args.toeplitz), max_cholesky_size(
        args.cholesky_size
    ), max_root_decomposition_size(args.sketch_size), cholesky_jitter(
        1e-3
    ), fast_pred_var(
        use_fast_pred_var 
    ), fast_pred_samples(
        True
    ):
        main(args)

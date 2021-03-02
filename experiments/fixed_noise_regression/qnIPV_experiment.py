import argparse
import time
import torch
import gpytorch

from botorch.models import FixedNoiseGP
from botorch.optim.fit import fit_gpytorch_torch
from botorch.optim import optimize_acqf
from botorch.sampling.samplers import SobolQMCNormalSampler
from gpytorch.kernels import GridInterpolationKernel, MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.settings import (
    max_cholesky_size,
    detach_test_caches,
    skip_logdet_forward,
    use_toeplitz,
    max_root_decomposition_size,
    fast_pred_var,
)
from pandas import DataFrame

from online_gp.mlls.batched_woodbury_marginal_log_likelihood import (
    BatchedWoodburyMarginalLogLikelihood,
)
from online_gp.settings import root_pred_var
from online_gp.models import FixedNoiseOnlineSKIGP, OnlineSKIBotorchModel
from online_gp.acquisition.active_learning import qNIPV

from data import prepare_data

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_init",
        type=int,
        help="(int) number of initial points",
        default=10,
    )
    parser.add_argument("--num_total", type=int, default=1000000)
    parser.add_argument(
        "--data_loc", type=str, default="../../datasets/malaria_df.hdf5"
    )
    parser.add_argument("--sketch_size", type=int, default=512)
    parser.add_argument("--cholesky_size", type=int, default=901)
    parser.add_argument("--output", type=str, default="./malaria_output.pt")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--toeplitz", action="store_true")
    parser.add_argument("--reset_training_data", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--model", type=str, choices=["exact", "wiski"])
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

def main(args):
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    init_dict, train_dict, test_dict = prepare_data(
        args.data_loc, args.num_init, args.num_total, test_is_year=False, seed=args.seed,
    )
    init_x, init_y, init_y_var = (
        init_dict["x"].to(device),
        init_dict["y"].to(device),
        init_dict["y_var"].to(device),
    )
    train_x, train_y, train_y_var = (
        train_dict["x"].to(device),
        train_dict["y"].to(device),
        train_dict["y_var"].to(device),
    )
    test_x, test_y, test_y_var = (
        test_dict["x"].to(device),
        test_dict["y"].to(device),
        test_dict["y_var"].to(device),
    )

    if args.model == "wiski":
        model = FixedNoiseOnlineSKIGP(
            init_x,
            init_y.view(-1, 1),
            init_y_var.view(-1, 1),
            GridInterpolationKernel(
                base_kernel=ScaleKernel(
                    MaternKernel(
                        ard_num_dims=2,
                        nu=0.5,
                        lengthscale_prior=GammaPrior(3.0, 6.0),
                    ),
                    outputscale_prior=GammaPrior(2.0, 0.15),
                ),
                grid_size=30,
                num_dims=2,
                grid_bounds=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            ),
            learn_additional_noise=False,
        ).to(device)

        mll_type = lambda x, y: BatchedWoodburyMarginalLogLikelihood(x, y, clear_caches_every_iteration=True)
    elif args.model == "exact":
        model = FixedNoiseGP(
            init_x,
            init_y.view(-1, 1),
            init_y_var.view(-1, 1),
            ScaleKernel(
                    MaternKernel(
                        ard_num_dims=2,
                        nu=0.5,
                        lengthscale_prior=GammaPrior(3.0, 6.0),
                    ),
                    outputscale_prior=GammaPrior(2.0, 0.15),
            ),
        ).to(device)   
        mll_type = ExactMarginalLogLikelihood

    mll = mll_type(model.likelihood, model)

    print("---- Fitting initial model ----")
    start = time.time()
    model.train()
    model.zero_grad()
    # with max_cholesky_size(args.cholesky_size), skip_logdet_forward(True), \
    #       use_toeplitz(args.toeplitz), max_root_decomposition_size(args.sketch_size):
    fit_gpytorch_torch(mll, options={"lr": 0.1, "maxiter": 1000})
    end = time.time()
    print("Elapsed fitting time: ", end - start)
    print("Named parameters: ", list(model.named_parameters()))

    print("--- Now computing initial RMSE")
    model.eval()
    with gpytorch.settings.skip_posterior_variances(True):
        test_pred = model(test_x)
        pred_rmse = ((test_pred.mean - test_y)**2).mean().sqrt()

    print("---- Initial RMSE: ", pred_rmse.item())

    all_outputs = []
    start_ind = init_x.shape[0]
    end_ind = int(start_ind + args.batch_size)
    for step in range(args.num_steps):
        if step > 0 and step % 25 == 0:
            print("Beginning step ", step)

        total_time_step_start = time.time()

        if step > 0:
            print("---- Fitting model ----")
            start = time.time()
            model.train()
            model.zero_grad()
            mll = mll_type(model.likelihood, model)
            # with skip_logdet_forward(True), max_root_decomposition_size(
            #         args.sketch_size
            #     ), max_cholesky_size(args.cholesky_size), use_toeplitz(
            #         args.toeplitz
            #     ):
            fit_gpytorch_torch(mll, options={"lr": 0.01 * (0.99**step), "maxiter": 300})
     
            model.zero_grad()
            end = time.time()
            print("Elapsed fitting time: ", end - start)
            print("Named parameters: ", list(model.named_parameters()))

        if not args.random:
            if args.model == "wiski":
                botorch_model = OnlineSKIBotorchModel(model = model)
            else:
                botorch_model = model

            # qmc_sampler = SobolQMCNormalSampler(num_samples=4)

            bounds = torch.stack([torch.zeros(2), torch.ones(2)]).to(device)
            qnipv = qNIPV(
                model=botorch_model, 
                mc_points=test_x, 
                # sampler=qmc_sampler,
            )

            #with use_toeplitz(args.toeplitz), root_pred_var(True), fast_pred_var(True):
            candidates, acq_value = optimize_acqf(
                acq_function=qnipv,
                bounds=bounds,
                q=args.batch_size,
                num_restarts=1,
                raw_samples=10,  # used for intialization heuristic
                options={"batch_limit": 5, "maxiter": 200},
            )
        else:
            candidates = torch.rand(args.batch_size, train_x.shape[-1], device = device, dtype=train_x.dtype)
            acq_value = torch.zeros(1)
            model.eval()
            _ = model(test_x[:10]) # to init caches

        print("---- Finished optimizing; now querying dataset ---- ")
        with torch.no_grad():
            covar_dists = model.covar_module(candidates, train_x)
            nearest_points = covar_dists.evaluate().argmax(dim=-1)
            new_x = train_x[nearest_points]
            new_y = train_y[nearest_points]
            new_y_var = train_y_var[nearest_points]

            todrop = torch.tensor([x in nearest_points for x in range(train_x.shape[0])])
            train_x, train_y, train_y_var = train_x[~todrop], train_y[~todrop], train_y_var[~todrop]
            print("New train_x shape", train_x.shape)
            print("--- Now updating model with simulator ----")
            model = model.condition_on_observations(X=new_x, Y=new_y.view(-1,1), noise=new_y_var.view(-1,1))

        print("--- Now computing updated RMSE")
        model.eval()
        # with gpytorch.settings.fast_pred_var(True), \
        #     detach_test_caches(True), \
        #     max_root_decomposition_size(args.sketch_size), \
        #     max_cholesky_size(args.cholesky_size), \
        #     use_toeplitz(args.toeplitz), root_pred_var(True):
        test_pred = model(test_x)
        pred_rmse = ((test_pred.mean.view(-1) - test_y.view(-1))**2).mean().sqrt()
        pred_avg_variance = test_pred.variance.mean()
                
        total_time_step_elapsed_time = time.time() - total_time_step_start
        step_output_list = [total_time_step_elapsed_time, acq_value.item(), pred_rmse.item(), pred_avg_variance.item()]
        print("Step RMSE: ", pred_rmse)
        all_outputs.append(step_output_list)
        
        start_ind = end_ind
        end_ind = int(end_ind + args.batch_size)

    output_dict = {
        "model_state_dict": model.cpu().state_dict(),
        "queried_points": {'x': model.cpu().train_inputs[0], 'y': model.cpu().train_targets},
        "results": DataFrame(all_outputs)
    }
    torch.save(output_dict, args.output)


if __name__ == "__main__":
    args = parse()
    with fast_pred_var(True), \
            use_toeplitz(args.toeplitz), \
            detach_test_caches(True), \
            max_cholesky_size(args.cholesky_size), \
            max_root_decomposition_size(args.sketch_size), \
            root_pred_var(True):
        main(args)

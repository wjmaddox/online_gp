import argparse
import math
import time
import torch
import pandas as pd
import numpy as np

from botorch.models import FixedNoiseGP
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.kernels import GridInterpolationKernel, MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import (
    max_cholesky_size,
    detach_test_caches,
    skip_logdet_forward,
    use_toeplitz,
)

from online_gp.models.batched_fixed_noise_online_gp import FixedNoiseOnlineSKIGP
from online_gp.mlls.batched_woodbury_marginal_log_likelihood import (
    BatchedWoodburyMarginalLogLikelihood,
)

from data import prepare_data

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_init",
        type=int,
        help="(int) number of initial points",
        default=500,
    )
    parser.add_argument("--num_total", type=int, default=5000)
    parser.add_argument(
        "--data_loc", type=str, default="../../datasets/malaria_df.hdf5"
    )
    parser.add_argument("--cholesky_size", type=int, default=800)
    parser.add_argument("--output", type=str, default="./malaria_output.pt")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--toeplitz", action="store_true")
    parser.add_argument("--reset_training_data", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()

def main(args):
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    init_dict, train_dict, test_dict = prepare_data(
        args.data_loc, args.num_init, args.num_total
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

    covar_module = ScaleKernel(
        MaternKernel(
            ard_num_dims=2,
            nu=0.5,
            lengthscale_prior=GammaPrior(3.0, 6.0),
        ),
        outputscale_prior=GammaPrior(2.0, 0.15),
    )
    if not args.exact:
        covar_module = GridInterpolationKernel(
            base_kernel=covar_module,
            grid_size=30,
            num_dims=2,
            grid_bounds=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        )
    model = FixedNoiseGP(
        init_x,
        init_y.view(-1, 1),
        init_y_var.view(-1, 1),
        covar_module=covar_module,
    ).to(device)
    model.mean_module = ZeroMean()

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    print("---- Fitting initial model ----")
    start = time.time()
    with skip_logdet_forward(True), use_toeplitz(args.toeplitz):
        fit_gpytorch_torch(mll, options={"lr": 0.1, "maxiter": 1000})
    end = time.time()
    print("Elapsed fitting time: ", end - start)

    model.zero_grad()
    model.eval()

    print("--- Generating initial predictions on test set ----")
    start = time.time()
    with detach_test_caches(True), max_cholesky_size(args.cholesky_size), use_toeplitz(args.toeplitz):
        pred_dist = model(train_x)

        pred_mean = pred_dist.mean.detach()
        # pred_var = pred_dist.variance.detach()
    end = time.time()
    print("Elapsed initial prediction time: ", end - start)

    rmse_initial = ((pred_mean.view(-1) - train_y.view(-1)) ** 2).mean().sqrt()
    print("Initial RMSE: ", rmse_initial.item())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    mll_time_list = []
    rmse_list = []
    for i in range(500, train_x.shape[0]):
        model.zero_grad()
        model.train()

        start = time.time()
        with skip_logdet_forward(True), max_cholesky_size(args.cholesky_size), use_toeplitz(args.toeplitz):
            loss = -mll(model(*model.train_inputs), model.train_targets).sum()

        loss.backward()
        mll_time = start - time.time()

        optimizer.step()
        model.zero_grad()
        optimizer.zero_grad()
        start = time.time()
        if not args.reset_training_data:
            with torch.no_grad():
                model.eval()
                model.posterior(train_x[i].unsqueeze(0))
                model = model.condition_on_observations(
                    X=train_x[i].unsqueeze(0),
                    Y=train_y[i].view(1, 1),
                    noise=train_y_var[i].view(-1, 1),
                )
        else:
            model.set_train_data(train_x[:i], train_y[:i], strict=False)
            model.likelihood.noise = train_y_var[:i].t()

        fantasy_time = start - time.time()
        mll_time_list.append([-mll_time, -fantasy_time])

        if i % 25 == 0:
            start = time.time()
            model.eval()
            model.zero_grad()

            with detach_test_caches(), max_cholesky_size(
                10000
            ):
                pred_dist = model(train_x)
            end = time.time()

            rmse = (
                ((pred_dist.mean - train_y.view(-1)) ** 2).mean().sqrt().item()
            )
            rmse_list.append([rmse, end - start])
            print("Current RMSE: ", rmse)
            #print(
            #    "Outputscale: ", model.covar_module.base_kernel.raw_outputscale
            #)
            #print(
            #    "Lengthscale: ",
            #    model.covar_module.base_kernel.base_kernel.raw_lengthscale,
            #)

            print("Step: ", i, "Train Loss: ", loss)
            optimizer.param_groups[0]["lr"] *= 0.9

    torch.save(
        {"training": mll_time_list, "predictions": rmse_list}, args.output
    )


if __name__ == "__main__":
    args = parse()
    main(args)

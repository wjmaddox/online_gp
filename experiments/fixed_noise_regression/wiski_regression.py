import argparse
import math
import time
import torch
import pandas as pd
import numpy as np

from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.kernels import GridInterpolationKernel, MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from gpytorch.settings import (
    max_cholesky_size,
    detach_test_caches,
    skip_logdet_forward,
    use_toeplitz,
    max_root_decomposition_size,
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
    parser.add_argument("--sketch_tolerance", type=float, default=0.01)
    parser.add_argument("--sketch_size", type=int, default=512)
    parser.add_argument("--cholesky_size", type=int, default=800)
    parser.add_argument("--output", type=str, default="./malaria_output.pt")
    parser.add_argument("--toeplitz", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()

def main(args):
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    init_dict, train_dict, test_dict = prepare_data(
        args.data_loc, args.num_init, args.num_total, test_is_year=False
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

    mll = BatchedWoodburyMarginalLogLikelihood(model.likelihood, model)

    print("---- Fitting initial model ----")
    start = time.time()
    with skip_logdet_forward(True), max_root_decomposition_size(args.sketch_size), use_toeplitz(args.toeplitz):
        fit_gpytorch_torch(mll, options={"lr": 0.1, "maxiter": 1000})
    end = time.time()
    print("Elapsed fitting time: ", end - start)

    model.zero_grad()
    model.eval()

    print("--- Generating initial predictions on test set ----")
    start = time.time()
    with detach_test_caches(True), max_root_decomposition_size(
        args.sketch_size
    ), max_cholesky_size(args.cholesky_size), use_toeplitz(args.toeplitz):
        pred_dist = model(test_x)

        pred_mean = pred_dist.mean.detach()
        # pred_var = pred_dist.variance.detach()
    end = time.time()
    print("Elapsed initial prediction time: ", end - start)

    rmse_initial = ((pred_mean.view(-1) - test_y.view(-1)) ** 2).mean().sqrt()
    print("Initial RMSE: ", rmse_initial.item())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    mll_time_list = []
    rmse_list = []
    for i in range(500, train_x.shape[0]):
        model.zero_grad()
        model.train()

        start = time.time()
        with skip_logdet_forward(True), max_root_decomposition_size(
            args.sketch_size
        ), max_cholesky_size(args.cholesky_size), use_toeplitz(
            args.toeplitz
        ):
            loss = -mll(model(train_x[:i]), train_y[:i]).sum()

        loss.backward()
        mll_time = start - time.time()

        optimizer.step()
        model.zero_grad()
        optimizer.zero_grad()
        start = time.time()
        with torch.no_grad():
            model.condition_on_observations(
                train_x[i].unsqueeze(0),
                train_y[i].view(1, 1),
                train_y_var[i].view(-1, 1),
                inplace=True,
            )
        fantasy_time = start - time.time()
        mll_time_list.append([-mll_time, -fantasy_time])

        if i % 25 == 0:
            start = time.time()
            model.eval()
            model.zero_grad()

            with detach_test_caches(), max_root_decomposition_size(args.sketch_size), max_cholesky_size(
               args.cholesky_size 
            ):
                pred_dist = model(test_x)
            end = time.time()

            rmse = (
                ((pred_dist.mean - test_y.view(-1)) ** 2).mean().sqrt().item()
            )
            rmse_list.append([rmse, end - start])
            print("Current RMSE: ", rmse)
            print(
                "Outputscale: ", model.covar_module.base_kernel.raw_outputscale
            )
            print(
                "Lengthscale: ",
                model.covar_module.base_kernel.base_kernel.raw_lengthscale,
            )

            print("Step: ", i, "Train Loss: ", loss)
            optimizer.param_groups[0]["lr"] *= 0.9

    torch.save(
        {"training": mll_time_list, "predictions": rmse_list}, args.output
    )


if __name__ == "__main__":
    args = parse()
    main(args)

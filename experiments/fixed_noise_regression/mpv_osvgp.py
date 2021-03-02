import argparse
import time
import torch
import gpytorch

from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import VariationalELBO
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

from online_gp.models import VariationalGPModel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.utils.grid import create_grid

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
    parser.add_argument("--output", type=str, default="./malaria_output.pt")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--lr_init", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--num_inducing", type=int, default=30)
    parser.add_argument("--learn_inducing", action="store_true")
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--acqf", type=str, default="random", choices=["random", "max_post_var", "max_test_var"])
    return parser.parse_args()

def fit_variational_model(mll, model, optimizer, X, Y, maxiter):
    old_loss = 1e10
    loss = 0.
    iter = 0.
    # botorch like stopping conditions
    while loss < old_loss and iter < maxiter:
        if iter > 0:
            old_loss = loss

        loss = -mll(model(X), Y).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if iter % 100 == 0:
            print("Iter: ", iter, "Loss: ", loss.item())
        iter += 1
    return model, loss


def generate_candidates(model, batch_size, device, maxiter=300):
    # force constraints
    initial_candidates = torch.rand(batch_size, 2, device=device)
    trans_candidates = torch.log(initial_candidates / (1. - initial_candidates) + 1e-6)
    trans_candidates.requires_grad_()
    optimizer = torch.optim.Adam([trans_candidates], lr = 0.01)

    model.eval()

    old_loss = 1e10
    loss = 0.
    iter = 0.
    # botorch like stopping conditions
    while loss < old_loss and iter < maxiter:  
        model.train()
        model.eval()

        old_loss = loss
        loss = -model(trans_candidates).variance.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()
        if iter % 100 == 0:
            print("Iter: ", iter, "Loss: ", loss.item())
        iter += 1

    return trans_candidates, loss

def main(args):
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    init_dict, train_dict, test_dict = prepare_data(
        args.data_loc, args.num_init, args.num_total, test_is_year=False, seed=args.seed
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

    likelihood = FixedNoiseGaussianLikelihood(noise=init_y_var)
    grid_pts = create_grid(grid_sizes = [30, 30], grid_bounds=torch.tensor([[0., 1.], [0., 1.]]))
    induc_points = torch.cat([x.reshape(-1,1) for x in torch.meshgrid(grid_pts)],dim=-1)
    
    model = VariationalGPModel(
        inducing_points = induc_points,
        mean_module = gpytorch.means.ZeroMean(),
        covar_module = ScaleKernel(
            MaternKernel(
                ard_num_dims=2,
                nu=0.5,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        ),
        streaming=True,
        likelihood=likelihood,
        beta=args.beta,
        learn_inducing_locations=args.learn_inducing,
    ).to(device)
    mll = VariationalELBO(model.likelihood, model, beta=args.beta, num_data=args.num_init)

    print("---- Fitting initial model ----")
    start = time.time()
    model.train()
    model.zero_grad()
    optimizer = torch.optim.Adam(model.parameters(), lr = 10 * args.lr_init)
    model, loss = fit_variational_model(mll, model, optimizer, init_x, init_y, maxiter=1000)
    end = time.time()
    print("Elapsed fitting time: ", end - start)

    print("--- Now computing initial RMSE")
    model.eval()
    with gpytorch.settings.skip_posterior_variances(True):
        test_pred = model(test_x)
        pred_rmse = ((test_pred.mean - test_y)**2).mean().sqrt()

    print("---- Initial RMSE: ", pred_rmse.item())

    all_outputs = []
    start_ind = init_x.shape[0]
    end_ind = int(start_ind + args.batch_size)

    current_x = init_x
    current_y = init_y
    current_y_var = init_y_var

    for step in range(args.num_steps):
        if step > 0 and step % 25 == 0:
            print("Beginning step ", step)

        total_time_step_start = time.time()

        if step > 0:
            print("---- Fitting model ----")
            start = time.time()
            model.train()
            model.zero_grad()
            model.likelihood = FixedNoiseGaussianLikelihood(current_y_var)
            mll = VariationalELBO(model.likelihood, model, beta=args.beta, num_data=args.num_init)
            optimizer = torch.optim.Adam(model.parameters(), lr = args.lr_init * 0.99**step)
            model, loss = fit_variational_model(mll, model, optimizer, current_x, current_y, maxiter=300)
     
            model.zero_grad()
            end = time.time()
            print("Elapsed fitting time: ", end - start)
            # print("Named parameters: ", list(model.named_parameters()))

        if args.acqf == "max_post_var" and not args.random:
            candidates, acq_value = generate_candidates(model, args.batch_size, device, maxiter=300)
        elif args.acqf == "max_test_var" and not args.random:
            model.eval()
            vals, inds = model(test_x).variance.sort()
            acq_value = vals[-args.batch_size:].mean().detach()
            candidates = test_x[inds[-args.batch_size:]]
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
            current_x = torch.cat((current_x, new_x), dim=0)
            current_y = torch.cat((current_y, new_y), dim=0)
            current_y_var = torch.cat((current_y_var, new_y_var), dim=0)

        print("--- Now computing updated RMSE")
        model.eval()
        test_pred = model(test_x)
        pred_rmse = ((test_pred.mean.view(-1) - test_y.view(-1))**2).mean().sqrt()
        pred_avg_variance = test_pred.variance.mean()
                
        total_time_step_elapsed_time = time.time() - total_time_step_start
        step_output_list = [total_time_step_elapsed_time, acq_value.item(), pred_rmse.item(), pred_avg_variance.item(), loss.item()]
        print("Step RMSE: ", pred_rmse)
        all_outputs.append(step_output_list)
        
        start_ind = end_ind
        end_ind = int(end_ind + args.batch_size)

    output_dict = {
        "model_state_dict": model.cpu().state_dict(),
        "queried_points": {'x': current_x, 'y': current_y},
        "results": DataFrame(all_outputs)
    }
    torch.save(output_dict, args.output)


if __name__ == "__main__":
    args = parse()
    args.random = args.acqf == "random"
    with fast_pred_var(True), \
            detach_test_caches(True):
        main(args)

import argparse
import torch

from botorch.acquisition import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qUpperConfidenceBound,
    qKnowledgeGradient,
    qMaxValueEntropy,
)
from botorch.sampling import IIDNormalSampler, SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize
from botorch.test_functions import (
    Ackley,
    Beale,
    Branin,
    Bukin,
    DropWave,
    DixonPrice,
    EggHolder,
    Griewank,
    Hartmann,
    HolderTable,
    Levy,
    Michalewicz,
    Rastrigin,
    Rosenbrock,
    StyblinskiTang,
)

# commented out means the apis are slightly different than how i've implemented
# mostly dim stuff
# TODO: make the apis work properly
# only tested for two d inputs. hartmann3....
test_functions_dict = {
    "Ackley": Ackley,
    #    "Beale": Beale,
    #    "Branin": Branin,
    #    "Bukin": Bukin,
    #    "DropWave": DropWave,
    "DixonPrice": DixonPrice,
    #    "EggHolder": EggHolder,
    "Griewank": Griewank,
    #    "Hartmann": Hartmann,
    #    "HolderTable": HolderTable,
    "Levy": Levy,
    "Michalewicz": Michalewicz,
    "Rastrigin": Rastrigin,
    "Rosenbrock": Rosenbrock,
    "StyblinskiTang": StyblinskiTang,
}


def prepare_function(args, device):
    test_function = test_functions_dict[args.function]
    test_function_class = test_function(dim=args.dim, negate=True, noise_std=0.0).to(
       device
    )

    class SubsetTestFunction(test_function):
        def forward(self, x):
            latent_y = test_function_class(x)
            noisy_y = latent_y + args.noise * torch.randn_like(latent_y)
            return noisy_y, latent_y

    test_function_instantiated = SubsetTestFunction(
        dim=args.dim, negate=True, noise_std=args.noise
    ).to(device)

    return test_function_instantiated


def prepare_acquisition_function(args, model_obj, train_x, train_y, bounds, step):
    if args.num_steps > 500:
        sampler = IIDNormalSampler(num_samples=256)
    else:
        sampler = SobolQMCNormalSampler(num_samples=256)
    if args.acqf == "ei":
        acqf = qExpectedImprovement(
            model=model_obj, best_f=train_y.max(), sampler=sampler,
        )
    elif args.acqf == "ucb":
        acqf = qUpperConfidenceBound(model=model_obj, beta=0.9 ** step)
    elif args.acqf == "nei":
        acqf = qNoisyExpectedImprovement(
            model=model_obj, X_baseline=train_x, sampler=sampler
        )
    elif args.acqf == "kg":
        acqf = qKnowledgeGradient(
            model=model_obj,
            sampler=sampler,
            num_fantasies=None,
            current_value=train_y.max(),
        )
    elif args.acqf == "mves":
        candidate_set = torch.rand(10000, bounds.size(0), device=bounds.device)
        candidate_set = bounds[..., 0] + (bounds[..., 1] - bounds[..., 0]) * candidate_set
        acqf = qMaxValueEntropy(
            model=model_obj, candidate_set=candidate_set, train_inputs=train_x,
        )

    return acqf


def initialize_random_data(test_function, device, n=10):
    init_x_cube = torch.rand(n, test_function.dim, device=device)
    init_x = (
        test_function.bounds[1] - test_function.bounds[0]
    ) * init_x_cube + test_function.bounds[0]
    init_y_unstandardized, latent_y = test_function(init_x)
    init_y_unstandardized = init_y_unstandardized.view(-1, 1)
    latent_y = latent_y.view(-1, 1)
    y_mean = init_y_unstandardized.mean(dim=0)
    y_std = init_y_unstandardized.std(dim=0)

    init_y = (init_y_unstandardized - y_mean) / y_std
    return init_x_cube, init_y, {"mean": y_mean, "std": y_std}, latent_y


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", type=str, default="Ackley")
    parser.add_argument(
        "--num_init", type=int, help="(int) number of initial points", default=10,
    )
    parser.add_argument("--sketch_size", type=int, default=512)
    parser.add_argument("--cholesky_size", type=int, default=901)
    parser.add_argument("--output", type=str, default="./malaria_output.pt")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--toeplitz", action="store_true")
    parser.add_argument("--reset_training_data", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--model", type=str, choices=["exact", "wiski", "osvgp"])
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--dim", default=2, type=int)
    parser.add_argument("--noise", default=0.1, type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_exact", action="store_true")
    parser.add_argument("--fixed_noise", action="store_true")
    parser.add_argument(
        "--acqf", type=str, choices=["ei", "nei", "ucb", "kg", "mves"], default="ucb"
    )
    return parser.parse_args()


def optimize_acqf_and_get_observation(
    acq_func, bounds, test_function_bounds, batch_size, test_function
):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=10,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = candidates.detach()
    new_x_unbounded = (
        new_x * (test_function_bounds[1] - test_function_bounds[0])
        + test_function_bounds[0]
    )
    new_obj, new_latent = test_function(new_x_unbounded)
    new_obj, new_latent = new_obj.view(-1, 1), new_latent.view(-1, 1)
    return new_x, new_obj, new_latent

# _Kernel Interpolation for Scalable Online Gaussian Processes_

This repository contains a gpytorch implementation of WISKI (Woodbury Inversion with SKI) from the paper 

[Kernel Interpolation for Scalable Online Gaussian Processes](https://arxiv.org/abs/2103.01454)

by Samuel Stanton, Wesley J. Maddox, Ian Delbridge, Andrew Gordon Wilson

:tumbler_glass:

## Introduction

While Gaussian processes are the gold standard for calibration and predictive performance in many settings, 
they scale at least $\mathcal{O}(n),$ where $n$ is the number of data points. We show how to use [structured
kernel interpolation](https://arxiv.org/abs/1503.01057) (SKI) to efficiently reuse computations to produce constant
time (in $n$) updates to
the posterior distribution, while retaining the exact inference formulation (no variational objectives) of
Gaussian processes.

## Installation

To replicate our experiments, you'll need to simply install the package:
```
git clone https://github.com/wjmaddox/online_gp.git
cd online_gp
pip install -r requirements.txt
pip install -e .
```

## Exploration of Different Types of Online Approximations

We've included an exploration and tutorial of different types of online approximate Gaussian processes (WISKI,
[Online SVGPs](https://arxiv.org/abs/1705.07131), and [Online SGPR](https://arxiv.org/abs/1705.07131)) in [this notebook](notebooks/regression_viz_1D.ipynb). 
We'd highly encourage the reader to start there to understand the differences between types of data observed
in the streaming setting (whether iid data or time series formatted data).

## Streaming Regression and Classification Experiments

The UCI regression and classification experiments require an additional data storage package for logging:

```
git clone https://github.com/samuelstanton/upcycle.git
pip install -e upcycle/

```

These experiments use [Hydra](https://hydra.cc/docs/intro/) to manage configuration. 
Every field in the `config/*.yaml` files can be 
overridden from the command line.

### Regression
```
python experiments/regression.py
```
Important options
- model=(exact_gp_regression, svgp_regression, sgpr_regression, wiski_gp_regression)
- dataset=(skillcraft, powerplant, elevators, protein, 3droad)
- stem=(eye, linear, mlp)

### Classification
```
python experiments/classification.py
```
Important options
- model=(exact_gpd, svgp_bin, wiski_gpd)
- dataset=(banana, svm_guide_1)
- stem=(eye, linear, mlp)

### Logging
By default your experimental results will be saved
as csv files in `data/experiments/<exp_name>`. 
If you have an Amazon AWS command line interface (CLI) configured you can modify
`config/logger/s3.yaml` and use the option
`logger=s3` to log results in a specified S3 bucket.

## Bayesian Optimization and Active Learning

Our bayesian optimization and active learning experiments are built off of [Botorch](https://botorch.org) 
and use standard bayesian optimization loops as in their tutorials. 

### Bayesion Optimization

```
cd experiments/bayesopt/
python bayesopt.py --model=wiski --cuda --cholesky_size=1001 \
    --dim=3 --acqf=ucb --function=Ackley \
    --noise=4.0 --num_steps=1500 --batch_size=3 --seed=0 \
    --output=results.pt
```

### Active Learning

The malaria dataset can be downloaded from [here](https://wjmaddox.github.io/assets/data/malaria_df.hdf5). 
Use the `--data_loc` to load the file in from where you downloaded it.

```
cd experiments/active_learning/

#### wiski and exact experiments with qnIPV
python qnIPV_experiment.py --cuda --batch_size=6 --num_steps=500 --model=exact
python qnIPV_experiment.py --cuda --batch_size=6 --num_steps=500 --model=wiski

##### osvgp experiments with osvgp's minimum posterior variance
# random
python mpv_osvgp.py --cuda --batch_size=6 --num_steps=500 --seed=0 --acqf=random --lr_init=1e-4 --output=svgp_random.pt

# maximum posterior variance
python mpv_osvgp.py --cuda --batch_size=6 --num_steps=500 --seed=0 --acqf=max_post_var --lr_init=1e-4 --output=svgp_mpv.pt
```

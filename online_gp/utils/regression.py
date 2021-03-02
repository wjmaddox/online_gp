import torch
from torch.utils.data import TensorDataset, DataLoader


def evaluate(gp_regression, inputs, targets):
    gp_regression.eval()
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=1024)
    rmse, nll = 0, 0
    num_batches = len(dataloader)
    for input_batch, target_batch in dataloader:
        pred_mean, pred_var = gp_regression.predict(input_batch)
        target_batch = gp_regression._reshape_targets(target_batch)
        rmse += (pred_mean - target_batch).pow(2).mean().sqrt().item() / num_batches
        diag_dist = torch.distributions.Normal(pred_mean, pred_var.sqrt())
        nll += -diag_dist.log_prob(target_batch).mean().item() / num_batches
    return rmse, nll

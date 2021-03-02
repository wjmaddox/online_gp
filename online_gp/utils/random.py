import torch


def shuffle_tensors(*tensors):
    batch_size = tensors[0].size(0)
    assert all([tensor.size(0) == batch_size for tensor in tensors])
    perm_idxs = torch.randperm(batch_size)
    return [tensor[perm_idxs] for tensor in tensors]

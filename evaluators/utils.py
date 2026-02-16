import torch

def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))

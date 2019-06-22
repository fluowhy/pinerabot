import random
import os
import numpy as np
import torch


class ToDevice(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, x):
        return x[0].to(self.device), x[1].to(self.device), x[2].to(self.device)


def seed_everything(seed=1111):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sort_by_zeros(x, y, pad=0):
    # sort samples by inverse number of zeros (padded inputs)
    nz = (x == pad).sum(dim=1)
    _, ind = torch.sort(nz, descending=False)
    return x[ind], y[ind]


def compute_lengths(x, pad=0):
    return (x != pad).sum(1)
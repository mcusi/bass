import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Common functions to ensure differentiability of the generative model
softplus = nn.Softplus()
sigmoid = nn.Sigmoid()

def softbound(x, lower, upper, ramp):
    r = F.hardtanh((x-lower)/(upper-lower), 0, 1)
    return (
        r * (upper - F.softplus(upper-x, 1/ramp)) +
        (1-r) * (lower + F.softplus(x-lower, 1/ramp))
    )

def torch_inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))

def np_inv_softplus(x):
    return x + np.log(-np.expm1(-x))

def np_softplus(x):
    return np.log(1+np.exp(-np.abs(x))) + np.maximum(x,0)
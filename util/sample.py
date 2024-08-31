import random
import numpy as np
import torch

def manual_seed(seed=None):
    if seed is None:
        seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def rsample(dist, *args, **kwargs):
    return dist.rsample(*args, **kwargs)

def sample_nograd(dist, *args, **kwargs):
    return dist.sample(*args, **kwargs)

def sample_delta(dist, sample_shape=torch.Size([])):
    if hasattr(dist, "median"):
        x = dist.median()
    elif hasattr(dist, "mean"):
        x = dist.mean
    else:
        raise NotImplementedError()
    
    x = x.expand([*sample_shape, *x.shape])
    return x

def sample_delta_nograd(*args, **kwargs):
    return sample_delta(*args, **kwargs).detach()

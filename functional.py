import torch
import torch.nn.functional as F
import math

def softclamp_upper(x: torch.Tensor, upper: float):
    return upper - F.softplus(upper - x)

def softclamp(x: torch.Tensor, upper: float, lower: float):
    return lower + F.softplus(x - lower) - F.softplus(x - upper)

def cosine_schedule(base, end, T):
    return end - (end - base) * ((torch.arange(0, T, 1) * math.pi / T).cos() + 1) / 2
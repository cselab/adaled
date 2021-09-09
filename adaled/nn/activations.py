# Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
import torch
import torch.nn as nn

from typing import Dict

__all__ = ['TanhPlus', 'make_activation']

class TanhPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 0.5 + self.tanh(x) * 0.5


class SoftplusPlusEpsilon(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(x) + 1e-6


torch_activations: Dict[str, nn.Module] = {
    'none': nn.Identity,
    'relu': nn.ReLU,
    'selu': nn.SELU,
    'elu': nn.ELU,
    'celu': nn.CELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'identity': nn.Identity,
    'tanhplus': TanhPlus,
    'solfplus': nn.Softplus,
    'softpluseps': SoftplusPlusEpsilon,
}

def make_activation(name: str):
    return torch_activations[name]()

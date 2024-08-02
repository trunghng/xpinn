from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np

from utils import ftensor


def mlp(sizes: List[int],
        activation: nn.Module,
        output_activation: nn.Module):
    layers = []
    for i in range(len(sizes) - 1):
        activation_ = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), activation_()]
    return nn.Sequential(*layers)


class PINN(nn.Module):

    def __init__(self, layers: List[int], activation: nn.Module) -> None:
        super().__init__()
        self.network = mlp(layers, activation, nn.Identity)


    def forward(self, inputs: List[torch.Tensor]):
        u = torch.cat(inputs, dim=-1)
        return self.network(u)


class XPINN(nn.Module):

    def __init__(self,
                layers_list: List[List[int]],
                activation_list: List[nn.Module]):
        super().__init__()
        self.subnets = nn.ModuleList([PINN(l, a) for l, a in zip(layers_list, activation_list)])


    def predict(self, test_data: List[Tuple[np.ndarray, np.ndarray]]):
        u_preds = []
        with torch.no_grad():
            for subnet, (x_test, y_test) in zip(self.subnets, test_data):
                u_preds.append(subnet([ftensor(x_test), ftensor(y_test)]))

        return u_preds

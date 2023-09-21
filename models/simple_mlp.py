from typing import List

import torch.nn as nn
from torch import Tensor

from models.init_weights import init_weights


class MLP(nn.Module):
    def __init__(
            self,
            in_dim: int = 2,
            out_dim: int = 1,
            dim: int = 32,
            dim_mults: List[int] = (1, 1),
            with_bn: bool = False,
            init_type: str = 'normal',
    ):
        """ A simple MLP.

        Args:
            in_dim: Input dimension.
            out_dim: Output dimension.
            dim: Base dimension.
            dim_mults: Multiplies of dimensions.
            with_bn: Use batch normalization.
            init_type: Type of weight initialization.

        """
        super().__init__()

        self.first_layer = nn.Linear(in_dim, dim * dim_mults[0])
        self.layers = nn.ModuleList([])
        for i in range(len(dim_mults) - 1):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(dim * dim_mults[i]) if i > 0 and with_bn else nn.Identity(),
                nn.LeakyReLU(0.2),
                nn.Linear(dim * dim_mults[i], dim * dim_mults[i+1])
            ))
        self.last_layer = nn.Sequential(
            nn.BatchNorm2d(dim * dim_mults[-1]) if with_bn else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Linear(dim * dim_mults[-1], out_dim),
        )
        self.apply(init_weights(init_type))

    def forward(self, X: Tensor):
        X = self.first_layer(X)
        for layer in self.layers:
            X = layer(X)
        X = self.last_layer(X)
        return X


def _test():
    import torch
    mlp = MLP()
    x = torch.randn(10, 2)
    out = mlp(x)
    print(out.shape)


if __name__ == '__main__':
    _test()

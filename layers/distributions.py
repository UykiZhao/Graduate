# -*- coding: utf-8 -*-
"""概率分布模块。"""

from typing import Tuple

import torch
import torch.nn as nn


class GaussianDiagonal(nn.Module):
    """对角协方差多元高斯分布。"""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc_mean = nn.Linear(input_dim, output_dim)
        self.fc_logvar = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x).clamp(min=-20.0, max=5.0)
        return mean, logvar


class ReparameterizedNormal(nn.Module):
    """重参数化正态分布采样器。"""

    def forward(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std



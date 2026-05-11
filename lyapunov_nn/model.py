from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class StructuredLyapunovNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Iterable[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.backbone = nn.Sequential(*layers)

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基础项保证在 0 附近有平方项（提高正定性倾向），
        # residual 用于学习比平方项更灵活的形状，但最终通过平方确保输出非负。
        base = x.pow(2).sum(dim=-1, keepdim=True)
        zero = torch.zeros_like(x)
        residual = self.raw(x) - self.raw(zero)
        return base + residual.pow(2)


class FreeLyapunovNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Iterable[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(x)
        return self.backbone(x) - self.backbone(zero)


class QuadraticLyapunovModel(nn.Module):
    def __init__(self, input_dim: int, min_diag: float = 1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.min_diag = min_diag
        self.tril_params = nn.Parameter(torch.zeros(input_dim, input_dim))

    def lyapunov_matrix(self) -> torch.Tensor:
        # 通过下三角矩阵参数化 P = L L^T，且用 softplus 保证对角元素为正，进而使 P 更倾向于正定
        tril = torch.tril(self.tril_params)
        diag_idx = torch.arange(self.input_dim, device=tril.device)
        tril = tril.clone()
        tril[diag_idx, diag_idx] = torch.nn.functional.softplus(tril[diag_idx, diag_idx]) + self.min_diag
        p = tril @ tril.T
        return p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.lyapunov_matrix()
        quadratic = torch.einsum("bi,ij,bj->b", x, p, x)
        return quadratic.unsqueeze(-1)

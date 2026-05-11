from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class StableCubicSystem:
    """简单的 2D 非线性系统示例，用于原型的训练与演示。

    方程形式为：dx1/dt = -x1 + 0.5 x2 - x1^3
             dx2/dt = -0.5 x1 - x2 - x2^3
    主要用于展示 NN 找 Lyapunov 函数的流程。
    """

    name: str = "stable_cubic_2d"
    state_dim: int = 2

    def dynamics(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0]
        x2 = x[..., 1]

        dx1 = -x1 + 0.5 * x2 - x1.pow(3)
        dx2 = -0.5 * x1 - x2 - x2.pow(3)

        return torch.stack((dx1, dx2), dim=-1)


@dataclass
class LinearSystem:
    """线性系统：x' = A x

    这里把 A 作为属性存储，并在 dynamics 中以矩阵乘法形式返回 A x。
    """
    name: str
    A: torch.Tensor

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    def dynamics(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A.T


def parse_linear_matrix(raw: str, state_dim: int, device: str) -> torch.Tensor:
    # 将用户通过命令行传入的逗号分隔字符串解析为矩阵
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    expected = state_dim * state_dim
    if len(values) != expected:
        raise ValueError(
            f"linear_a expects {expected} values for a {state_dim}x{state_dim} matrix, got {len(values)}"
        )
    matrix = torch.tensor(values, dtype=torch.float32, device=device).reshape(state_dim, state_dim)
    return matrix


def get_system(name: str, state_dim: int = 2, linear_a: str | None = None, device: str = "cpu"):
    # 根据 name 返回对应的系统实例，目前支持两种：内置非线性示例和线性系统
    if name == "stable_cubic_2d":
        return StableCubicSystem()

    if name == "linear":
        default_a = "-1,0,0,-2" if state_dim == 2 else None
        raw = linear_a if linear_a is not None else default_a
        if raw is None:
            raise ValueError("For linear systems, provide --linear-a with row-major matrix values.")
        matrix = parse_linear_matrix(raw, state_dim, device)
        return LinearSystem(name="linear", A=matrix)

    raise ValueError(f"Unsupported system: {name}")

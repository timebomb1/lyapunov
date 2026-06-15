from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import importlib.util
from pathlib import Path

import torch


@dataclass
class StableCubicSystem:
    """简单的二维非线性系统示例，用于原型训练与演示。

    方程形式为：dx1/dt = -x1 + 0.5 x2 - x1^3
             dx2/dt = -0.5 x1 - x2 - x2^3
    主要用于展示神经网络搜索 Lyapunov 函数的流程。
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
    # 将命令行传入的逗号分隔字符串解析为矩阵
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    expected = state_dim * state_dim
    if len(values) != expected:
        raise ValueError(
            f"linear_a expects {expected} values for a {state_dim}x{state_dim} matrix, got {len(values)}"
        )
    matrix = torch.tensor(values, dtype=torch.float32, device=device).reshape(state_dim, state_dim)
    return matrix


def load_custom_system_from_file(filepath: str):
    """从 Python 文件动态加载自定义系统定义。
    
    用户需要在文件中定义一个名为 `system` 的全局对象（系统实例），
    或者定义一个名为 `create_system()` 的函数返回系统实例。
    
    系统对象需要包含：
    - dynamics(x: torch.Tensor) -> torch.Tensor：系统动力学函数
    - state_dim: int 或 @property：状态维数
    - name: str（可选）：系统名称
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"系统定义文件不存在: {filepath}")
    
    spec = importlib.util.spec_from_file_location("custom_system", filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {filepath}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # 尝试获取系统实例或工厂函数
    if hasattr(module, "system"):
        return module.system
    elif hasattr(module, "create_system") and callable(module.create_system):
        return module.create_system()
    else:
        raise AttributeError(
            f"自定义系统文件需要定义 'system' 实例或 'create_system()' 函数，但都未找到: {filepath}"
        )


def get_system(
    name: str | None = None,
    state_dim: int = 2,
    linear_a: str | None = None,
    custom_system_file: str | None = None,
    device: str = "cpu",
):
    """根据系统名称或自定义文件返回对应的系统实例。
    
    优先级：
    1. 如果提供 custom_system_file，从文件加载
    2. 如果 name 为已知类型，使用内置系统
    """
    if custom_system_file is not None:
        return load_custom_system_from_file(custom_system_file)
    
    if name is None:
        name = "stable_cubic_2d"
    
    # 根据系统名称返回对应的系统实例，目前支持内置非线性示例和线性系统
    if name == "stable_cubic_2d":
        return StableCubicSystem()

    if name == "linear":
        default_a = "-1,0,0,-2" if state_dim == 2 else None
        raw = linear_a if linear_a is not None else default_a
        if raw is None:
            raise ValueError("线性系统需要通过 --linear-a 提供矩阵 A 的行优先展开值。")
        matrix = parse_linear_matrix(raw, state_dim, device)
        return LinearSystem(name="linear", A=matrix)

    raise ValueError(f"不支持的系统名称: {name}")

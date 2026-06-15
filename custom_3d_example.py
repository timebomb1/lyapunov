"""3D 非线性系统示例，用于自动求解 Lyapunov 函数"""

from dataclasses import dataclass
import torch


@dataclass
class ThreeDimSystem:
    """3D 非线性系统
    
    方程：
        dx1/dt = -x1 + 0.1*x2*x3
        dx2/dt = -x2 - 0.1*x1*x3
        dx3/dt = -x3 - x1*x2
    """
    name: str = "three_dim_demo"
    state_dim: int = 3
    
    def dynamics(self, x: torch.Tensor) -> torch.Tensor:
        """计算 3D 系统的导数"""
        x1 = x[..., 0]
        x2 = x[..., 1]
        x3 = x[..., 2]
        
        dx1 = -x1 + 0.1 * x2 * x3
        dx2 = -x2 - 0.1 * x1 * x3
        dx3 = -x3 - x1 * x2
        
        return torch.stack((dx1, dx2, dx3), dim=-1)


# 使用工厂函数方式
def create_system():
    return ThreeDimSystem()


# 或直接定义系统实例
system = ThreeDimSystem()

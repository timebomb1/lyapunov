from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_v_and_dvdt(model, system, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 为了用自动微分计算 V 对输入的梯度，这里让 x 参与求导
    x = x.clone().detach().requires_grad_(True)
    v = model(x)
    # 计算 V 对 x 的梯度；create_graph=True 便于后续继续反向传播
    grad_v = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    fx = system.dynamics(x)
    # 通过链式法则计算 dV/dt = grad(V) · f(x)
    dvdt = (grad_v * fx).sum(dim=-1, keepdim=True)
    return v, dvdt


def lyapunov_objective(model, system, x: torch.Tensor, cfg) -> tuple[torch.Tensor, dict[str, float]]:
    v, dvdt = compute_v_and_dvdt(model, system, x)
    origin = torch.zeros(1, x.shape[-1], device=x.device, dtype=x.dtype)
    v0 = model(origin)
    # 损失项说明：
    # positivity_loss 惩罚 V(x) 过小的点
    # derivative_loss 惩罚 dV/dt 非负的点
    # origin_loss 确保 V(0) 接近 0
    positivity_loss = F.relu(cfg.positivity_margin - v).mean()
    derivative_loss = F.relu(dvdt + cfg.derivative_margin).mean()
    origin_loss = v0.pow(2).mean()
    scale_loss = 1e-3 * v.mean()

    total = (
        cfg.positivity_weight * positivity_loss
        + cfg.derivative_weight * derivative_loss
        + cfg.origin_weight * origin_loss
        + scale_loss
    )

    metrics = {
        "v_mean": float(v.mean().detach().cpu()),
        "v_min": float(v.min().detach().cpu()),
        "dvdt_mean": float(dvdt.mean().detach().cpu()),
        "positivity_loss": float(positivity_loss.detach().cpu()),
        "derivative_loss": float(derivative_loss.detach().cpu()),
        "origin_loss": float(origin_loss.detach().cpu()),
    }
    return total, metrics

from __future__ import annotations

from dataclasses import dataclass

import torch

from .losses import compute_v_and_dvdt


@dataclass
class VerificationReport:
    v_min: float
    v_mean: float
    dvdt_max: float
    dvdt_mean: float
    positivity_violation_rate: float
    derivative_violation_rate: float
    origin_value: float


def sample_states(num_samples: int, state_dim: int, radius: float, device: str, generator=None) -> torch.Tensor:
    # 在给定半径的超立方体内做均匀采样，作为训练与验证的状态点集合
    samples = torch.empty(num_samples, state_dim, device=device)
    samples.uniform_(-radius, radius, generator=generator)
    return samples


def evaluate_candidate(model, system, cfg, num_samples: int | None = None) -> VerificationReport:
    samples = num_samples or cfg.eval_samples
    x = sample_states(samples, cfg.state_dim, cfg.radius, cfg.device)
    v, dvdt = compute_v_and_dvdt(model, system, x)

    positivity_violation = (v < cfg.positivity_margin).float().mean().item()
    derivative_violation = (dvdt > -cfg.derivative_margin).float().mean().item()
    origin = torch.zeros(1, cfg.state_dim, device=cfg.device)
    origin_value = float(model(origin).item())

    return VerificationReport(
        v_min=float(v.min().item()),
        v_mean=float(v.mean().item()),
        dvdt_max=float(dvdt.max().item()),
        dvdt_mean=float(dvdt.mean().item()),
        positivity_violation_rate=positivity_violation,
        derivative_violation_rate=derivative_violation,
        origin_value=origin_value,
    )


def collect_hard_examples(model, system, cfg, num_samples: int, topk: int) -> torch.Tensor:
    x = sample_states(num_samples, cfg.state_dim, cfg.radius, cfg.device)
    v, dvdt = compute_v_and_dvdt(model, system, x)
    # 违例得分用于挑选最难样本（positivity 与 derivative 的违例之和）
    violation_score = torch.relu(cfg.positivity_margin - v).squeeze(-1) + torch.relu(dvdt + cfg.derivative_margin).squeeze(-1)
    topk = min(topk, x.shape[0])
    _, indices = torch.topk(violation_score, k=topk)
    return x[indices].detach()

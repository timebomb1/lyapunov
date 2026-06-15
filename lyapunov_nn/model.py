from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def _tensor_to_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


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
        # 基础项保证在原点附近有平方项，提升正定性倾向
        # residual 学习更灵活的形状，但通过平方保证输出非负
        base = x.pow(2).sum(dim=-1, keepdim=True)
        zero = torch.zeros_like(x)
        residual = self.raw(x) - self.raw(zero)
        return base + residual.pow(2)

    def export_lyapunov_description(self) -> dict[str, object]:
        return {
            "type": "structured_nn",
            "formula": "V(x) = ||x||^2 + (g(x) - g(0))^2",
        }


def _generate_exponents(state_dim: int, max_degree: int) -> list[tuple[int, ...]]:
    exponents: list[tuple[int, ...]] = []

    def build_for_total_degree(total_degree: int, index: int, prefix: list[int]) -> None:
        if index == state_dim - 1:
            exponents.append(tuple(prefix + [total_degree]))
            return

        for value in range(total_degree + 1):
            build_for_total_degree(total_degree - value, index + 1, prefix + [value])

    for total_degree in range(max_degree + 1):
        build_for_total_degree(total_degree, 0, [])

    return exponents


def _format_monomial(exponents: tuple[int, ...]) -> str:
    factors: list[str] = []
    for index, power in enumerate(exponents, start=1):
        if power == 0:
            continue
        if power == 1:
            factors.append(f"x{index}")
        else:
            factors.append(f"x{index}^{power}")
    return "1" if not factors else "*".join(factors)


def _build_polynomial_features(x: torch.Tensor, exponents: list[tuple[int, ...]]) -> torch.Tensor:
    features: list[torch.Tensor] = []
    for exp in exponents:
        feature = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        for dim, power in enumerate(exp):
            if power > 0:
                feature = feature * x[:, dim].pow(power)
        features.append(feature)
    return torch.stack(features, dim=-1)


def _round_float(value: float, digits: int = 6) -> float:
    return float(round(value, digits))
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

    def export_lyapunov_description(self) -> dict[str, object]:
        return {
            "type": "free_nn",
            "formula": "V(x) = g(x) - g(0)",
        }


class QuadraticLyapunovModel(nn.Module):
    def __init__(self, input_dim: int, min_diag: float = 1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.min_diag = min_diag
        self.tril_params = nn.Parameter(torch.zeros(input_dim, input_dim))

    def lyapunov_matrix(self) -> torch.Tensor:
        # 通过下三角矩阵参数化 P = L L^T，并用 softplus 保证对角元素为正
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

    def export_lyapunov_description(self) -> dict[str, object]:
        p = self.lyapunov_matrix()
        return {
            "type": "quadratic",
            "formula": "V(x) = x^T P x",
            "P": _tensor_to_list(p),
        }


def export_constructed_lyapunov_function(
    model: nn.Module,
    state_dim: int,
    device: str,
    radius: float,
    num_samples: int = 4096,
    max_degree: int = 4,
) -> dict[str, object]:
    if hasattr(model, "lyapunov_matrix"):
        return model.export_lyapunov_description()

    sample_count = max(2 * num_samples, 1)
    samples = torch.empty(sample_count, state_dim, device=device)
    samples.uniform_(-radius, radius)

    with torch.no_grad():
        targets = model(samples).squeeze(-1)

    exponents = _generate_exponents(state_dim, max_degree)
    features = _build_polynomial_features(samples, exponents)
    solution = torch.linalg.lstsq(features, targets.unsqueeze(-1)).solution.squeeze(-1)

    terms: list[dict[str, object]] = []
    parts: list[str] = []
    for coeff, exponent in zip(solution.tolist(), exponents):
        coeff_value = _round_float(coeff)
        if abs(coeff_value) < 1e-8:
            continue
        term = _format_monomial(exponent)
        terms.append({"coefficient": coeff_value, "monomial": term})
        parts.append(f"{coeff_value}*{term}" if term != "1" else f"{coeff_value}")

    formula = "V(x) = " + (" + ".join(parts) if parts else "0")
    fit_error = torch.mean((features @ solution - targets) ** 2).sqrt().item()

    return {
        "type": "polynomial_surrogate",
        "source": "trained_nonlinear_lyapunov_network",
        "formula": formula,
        "degree": max_degree,
        "terms": terms,
        "fit_rmse": _round_float(fit_error),
    }

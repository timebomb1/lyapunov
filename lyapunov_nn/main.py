from __future__ import annotations

import argparse
from pathlib import Path
import json
import torch

from .config import ExperimentConfig
from .systems import get_system
from .train import save_artifacts, search_best_model, make_model
from .model import export_constructed_lyapunov_function


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="结构化神经 Lyapunov 原型")
    parser.add_argument("--system-name", type=str, default="stable_cubic_2d", choices=("stable_cubic_2d", "linear"))
    parser.add_argument("--custom-system-file", type=str, default=None, help="自定义非线性系统的 Python 文件路径（优先于 --system-name）")
    parser.add_argument("--state-dim", type=int, default=2)
    parser.add_argument(
        "--linear-a",
        type=str,
        default=None,
        help="线性系统矩阵 A 的行优先展开值，例如 '-1,0,0,-2'",
    )
    parser.add_argument("--model-kind", type=str, default=None, choices=("structured_nn", "quadratic"))
    parser.add_argument("--output-dir", type=str, default="runs/demo")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_kind = args.model_kind
    
    # 根据系统名称或自定义文件创建系统实例
    system = get_system(
        name=args.system_name,
        state_dim=args.state_dim,
        linear_a=args.linear_a,
        custom_system_file=args.custom_system_file,
        device=args.device,
    )
    
    if model_kind is None:
        # 如果未指定 model_kind，根据系统类型自动选择
        system_name = system.name if hasattr(system, "name") else args.system_name
        model_kind = "quadratic" if system_name == "linear" else "structured_nn"

    # 获取系统的真实名称（自定义系统的名称优先）
    actual_system_name = system.name if hasattr(system, "name") else args.system_name

    cfg = ExperimentConfig(
        system_name=actual_system_name,
        model_kind=model_kind,
        output_dir=args.output_dir,
        state_dim=system.state_dim,
        epochs=args.epochs,
        restarts=args.restarts,
        seed=args.seed,
        device=args.device,
        radius=args.radius,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    result = search_best_model(system, cfg)
    save_artifacts(result, system, cfg)

    output_dir = Path(cfg.output_dir)
    model = make_model(cfg.state_dim, cfg.hidden_sizes, cfg.device, model_kind=cfg.model_kind)
    model.load_state_dict(result.state_dict)
    model.eval()

    report = result.report
    candidate = export_constructed_lyapunov_function(
        model,
        state_dim=cfg.state_dim,
        device=cfg.device,
        radius=cfg.radius,
    )

    system_payload: dict[str, object] = {"name": cfg.system_name, "state_dim": cfg.state_dim}
    if hasattr(system, "A"):
        system_payload["A"] = system.A.detach().cpu().tolist()

    payload = {
        "seed": result.seed,
        "best_epoch": result.best_epoch,
        "best_loss": result.best_loss,
        "system": system_payload,
        "candidate_lyapunov": candidate,
        "verification": {
            "v_min": report.v_min,
            "v_mean": report.v_mean,
            "dvdt_max": report.dvdt_max,
            "dvdt_mean": report.dvdt_mean,
            "positivity_violation_rate": report.positivity_violation_rate,
            "derivative_violation_rate": report.derivative_violation_rate,
            "origin_value": report.origin_value,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with (output_dir / "lyapunov_function.json").open("w", encoding="utf-8") as f:
        json.dump(candidate, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

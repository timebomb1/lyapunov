from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch

from .losses import lyapunov_objective
from .verify import collect_hard_examples, sample_states, evaluate_candidate


@dataclass
class TrainingResult:
    seed: int
    best_epoch: int
    best_loss: float
    history: list[dict[str, float]]
    report: object
    state_dict: dict[str, torch.Tensor]


def set_seed(seed: int) -> None:
    # 固定随机数种子，确保可重复性
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(state_dim: int, hidden_sizes, device: str, model_kind: str = "structured_nn"):
    from .model import QuadraticLyapunovModel, StructuredLyapunovNet

    if model_kind == "quadratic":
        model = QuadraticLyapunovModel(state_dim)
    else:
        model = StructuredLyapunovNet(state_dim, hidden_sizes)
    return model.to(device)


def build_model(cfg):
    return make_model(cfg.state_dim, cfg.hidden_sizes, cfg.device, model_kind=cfg.model_kind)


def train_single_run(system, cfg, seed: int) -> TrainingResult:
    set_seed(seed)
    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    hard_examples = None
    best_state_dict = None
    best_loss = float("inf")
    best_epoch = 0
    patience = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        # 每次训练从采样器中取一批随机状态，混合 hard examples（困难样本）进行训练
        random_batch = sample_states(cfg.batch_size, cfg.state_dim, cfg.radius, cfg.device)
        if hard_examples is not None and hard_examples.numel() > 0:
            hard_count = int(cfg.batch_size * cfg.hard_example_ratio)
            hard_count = max(1, min(hard_count, hard_examples.shape[0]))
            random_count = cfg.batch_size - hard_count
            hard_idx = torch.randint(0, hard_examples.shape[0], (hard_count,), device=cfg.device)
            hard_batch = hard_examples[hard_idx]
            batch = torch.cat((random_batch[:random_count], hard_batch), dim=0)
        else:
            batch = random_batch

        optimizer.zero_grad(set_to_none=True)
        # 计算目标损失并反向传播，目标是同时减小正定性/导数违例
        loss, metrics = lyapunov_objective(model, system, batch, cfg)
        loss.backward()
        optimizer.step()

        history.append({"epoch": float(epoch), "loss": float(loss.detach().cpu()), **metrics})

        if epoch % cfg.log_interval == 0 or epoch == 1:
            model.eval()
            report = evaluate_candidate(model, system, cfg, num_samples=cfg.val_samples)
            score = report.derivative_violation_rate + report.positivity_violation_rate + report.origin_value
            if score < best_loss:
                best_loss = score
                best_epoch = epoch
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            hard_examples = collect_hard_examples(
                model,
                system,
                cfg,
                num_samples=cfg.train_samples,
                topk=cfg.hard_example_topk,
            )

            if patience >= cfg.early_stop_patience:
                break

    if best_state_dict is None:
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state_dict)
    model.to(cfg.device)
    model.eval()
    report = evaluate_candidate(model, system, cfg, num_samples=cfg.eval_samples)
    return TrainingResult(
        seed=seed,
        best_epoch=best_epoch,
        best_loss=best_loss,
        history=history,
        report=report,
        state_dict=best_state_dict,
    )


def search_best_model(system, cfg) -> TrainingResult:
    best_result: TrainingResult | None = None
    for offset in range(cfg.restarts):
        seed = cfg.seed + offset
        result = train_single_run(system, cfg, seed)
        if best_result is None or result.best_loss < best_result.best_loss:
            best_result = result
    assert best_result is not None
    return best_result


def save_artifacts(result: TrainingResult, system, cfg) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best_model.pt"
    torch.save(result.state_dict, checkpoint_path)

    history_path = output_dir / "history.pt"
    torch.save(result.history, history_path)

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    # 系统名称：默认是一个内置的二维非线性示例
    system_name: str = "stable_cubic_2d"
    # 模型类型：structured_nn 表示结构化神经网络候选，quadratic 表示二次型参数化
    model_kind: str = "structured_nn"
    state_dim: int = 2
    seed: int = 7
    device: str = "cpu"

    radius: float = 2.0
    train_samples: int = 512
    val_samples: int = 2048
    eval_samples: int = 8192

    hidden_sizes: tuple[int, ...] = (64, 64)
    lr: float = 1e-3
    epochs: int = 400
    restarts: int = 3
    batch_size: int = 256
    log_interval: int = 25
    early_stop_patience: int = 60

    positivity_margin: float = 1e-3
    derivative_margin: float = 1e-3
    positivity_weight: float = 1.0
    derivative_weight: float = 5.0
    origin_weight: float = 10.0
    hard_example_ratio: float = 0.3
    hard_example_topk: int = 64

    output_dir: str = "runs/demo"

    extra: dict[str, float] = field(default_factory=dict)

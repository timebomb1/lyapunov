from __future__ import annotations

from pathlib import Path

import torch

from .verify import sample_states


def maybe_plot_training_curve(history: list[dict[str, float]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not history:
        return

    epochs = [item["epoch"] for item in history]
    losses = [item["loss"] for item in history]
    derivative = [item["derivative_loss"] for item in history]
    positivity = [item["positivity_loss"] for item in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, losses, label="total")
    ax.plot(epochs, derivative, label="derivative")
    ax.plot(epochs, positivity, label="positivity")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def maybe_plot_landscape(model, system, cfg, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if cfg.state_dim != 2:
        return

    grid = torch.linspace(-cfg.radius, cfg.radius, cfg.plot_grid_size, device=cfg.device)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    with torch.no_grad():
        values = model(points).reshape(cfg.plot_grid_size, cfg.plot_grid_size).cpu().numpy()
        dynamics = system.dynamics(points).reshape(cfg.plot_grid_size, cfg.plot_grid_size, 2).cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(xx.cpu().numpy(), yy.cpu().numpy(), values, levels=30, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="V(x)")
    stride = max(1, cfg.plot_grid_size // 20)
    ax.quiver(
        xx.cpu().numpy()[::stride, ::stride],
        yy.cpu().numpy()[::stride, ::stride],
        dynamics[::stride, ::stride, 0],
        dynamics[::stride, ::stride, 1],
        color="white",
        alpha=0.7,
        linewidth=0.5,
    )
    ax.set_title("Learned Lyapunov landscape")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

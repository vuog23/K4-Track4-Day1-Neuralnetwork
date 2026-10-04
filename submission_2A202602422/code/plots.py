"""Plotting helpers for individual runs and experiment comparisons.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    cfg = result["cfg"]
    history = result["history"]
    epochs = history["epoch"]
    exp_id = cfg.get("exp_id", "experiment")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    if not epochs:
        for axis, label in zip(axes, ("Loss", "Validation metrics", "Gradient norm")):
            axis.set_title(label)
            axis.set_xlabel("Epoch")
            axis.text(0.5, 0.5, "No completed epochs\nRun diverged during training",
                      ha="center", va="center", transform=axis.transAxes)
        fig.suptitle(f"{exp_id}: training diverged before first epoch completed")
        fig.tight_layout()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="validation")
    axes[0].set_title("Loss (eval mode)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], marker="o", label="accuracy")
    if "val_macro_f1" in history:
        axes[1].plot(epochs, history["val_macro_f1"], marker="o", label="macro-F1")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    axes[2].plot(epochs, history["grad_norm"], marker="o", label="pre-clip norm")
    axes[2].set_title("Gradient norm")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Global L2 norm")
    axes[2].legend()

    best_epoch = result["summary"].get("best_epoch")
    if best_epoch in epochs:
        for axis in axes:
            axis.axvline(best_epoch, color="gray", linestyle="--", alpha=0.6)
    config_text = (
        f"{cfg.get('optimizer')} · lr={cfg.get('lr')} · batch={cfg.get('batch')} · "
        f"loss={cfg.get('loss')} · init={cfg.get('init')}"
    )
    fig.suptitle(f"{exp_id}: {config_text}")
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        raise ValueError("results must contain at least one run")
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        history = result["history"]
        if metric not in history:
            raise KeyError(f"{metric!r} is not in the run history")
        ax.plot(history["epoch"], history[metric], marker="o", label=result["cfg"].get("exp_id", "run"))
    ax.set_title(title or metric.replace("_", " ").title())
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " "))
    ax.legend()
    ax.grid(alpha=0.25)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

"""Plotting helpers — all functions write a PNG and return the file path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def hist_plot(
    values: np.ndarray,
    title: str,
    xlabel: str,
    out_path: str | Path,
    bins: int = 40,
) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=bins, color="#3a78c2", edgecolor="white", alpha=0.9)
    mean = float(np.mean(values))
    median = float(np.median(values))
    ax.axvline(mean, linestyle="--", color="black", label=f"mean={mean:.3f}")
    ax.axvline(median, linestyle=":", color="gray", label=f"median={median:.3f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
 

def per_cluster_bar(
    cluster_ids: np.ndarray,
    metric_values: np.ndarray,
    metric_name: str,
    out_path: str | Path,
) -> Path:
    """Bar chart of mean(metric) per cluster, ordered by cluster id."""
    out_path = Path(out_path)
    uniq = np.sort(np.unique(cluster_ids))
    means = np.array([metric_values[cluster_ids == c].mean() for c in uniq])

    fig, ax = plt.subplots(figsize=(max(8, len(uniq) * 0.35), 4))
    ax.bar(uniq, means, color="#4c9a4f", edgecolor="white")
    ax.set_xlabel("cluster id")
    ax.set_ylabel(f"mean {metric_name}")
    ax.set_title(f"Mean {metric_name} per cluster")
    ax.set_xticks(uniq)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def loss_curve(
    train_loss: list[float],
    val_loss: list[float],
    out_path: str | Path,
) -> Path:
    """Train/val loss curve from training history."""
    out_path = Path(out_path)
    epochs = np.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, label="train", color="#3a78c2")
    ax.plot(epochs, val_loss, label="val", color="#c23a3a")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstructor training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

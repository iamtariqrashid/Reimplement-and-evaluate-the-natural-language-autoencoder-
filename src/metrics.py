"""Reconstruction metrics: FVE, MSE, cosine similarity."""
from __future__ import annotations

import numpy as np


def mse_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean squared error averaged over feature dim, returned per sample."""
    return ((y_true - y_pred) ** 2).mean(axis=1)


def cosine_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Cosine similarity between matching rows."""
    num = (y_true * y_pred).sum(axis=1)
    den = (
        np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)
    ).clip(min=1e-12)
    return num / den


def fve_global(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of Variance Explained (global scalar).

    FVE = 1 - SS_res / SS_tot
        = 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)

    The mean is computed from y_true (typically the test set), matching the
    convention used in the Anthropic paper.
    """
    y_mean = y_true.mean(axis=0, keepdims=True)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_mean) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def fve_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-sample FVE using the same mean as the global metric.

    Each sample's variance contribution is computed against the dataset mean,
    so a single histogram entry is comparable across rows.
    """
    y_mean = y_true.mean(axis=0, keepdims=True)
    ss_res = ((y_true - y_pred) ** 2).sum(axis=1)
    ss_tot = ((y_true - y_mean) ** 2).sum(axis=1).clip(min=1e-12)
    return 1.0 - ss_res / ss_tot


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dict of headline metrics + per-sample arrays for plotting."""
    mse = mse_per_sample(y_true, y_pred)
    cos = cosine_per_sample(y_true, y_pred)
    fve_s = fve_per_sample(y_true, y_pred)
    return {
        "fve_global": fve_global(y_true, y_pred),
        "mse_mean": float(mse.mean()),
        "mse_median": float(np.median(mse)),
        "cosine_mean": float(cos.mean()),
        "cosine_median": float(np.median(cos)),
        "fve_sample_mean": float(fve_s.mean()),
        "fve_sample_median": float(np.median(fve_s)),
        "_per_sample": {
            "mse": mse,
            "cosine": cos,
            "fve": fve_s,
        },
    }

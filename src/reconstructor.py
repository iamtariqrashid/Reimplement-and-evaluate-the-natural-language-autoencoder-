"""Activation Reconstructor: MLP that maps explanation embeddings to activations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
class MLPReconstructor(nn.Module):
    """Small MLP: explanation_embedding -> reconstructed activation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (512, 1024),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# Training utilities                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_splits(
    n: int,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> SplitIndices:
    """Random sample-level train/val/test split. Reproducible via seed."""
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be in (0,1) and sum < 1.")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = idx[:n_test]
    val = idx[n_test : n_test + n_val]
    train = idx[n_test + n_val :]
    return SplitIndices(train=train, val=val, test=test)


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1


def train_reconstructor(
    model: MLPReconstructor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    early_stopping_patience: int,
    device: torch.device,
    log=None,
) -> tuple[dict, TrainHistory]:
    """Train the MLP; return (best_state_dict, history)."""
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    X_val_d, y_val_d = X_val.to(device), y_val.to(device)
    history = TrainHistory()
    best_state: dict | None = None
    patience = 0

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = total / max(n, 1)

        # ---- val ----
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_d), y_val_d).item()

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        if log is not None:
            log.info(
                "epoch %3d | train_mse=%.6f | val_mse=%.6f", epoch, train_loss, val_loss
            )

        # ---- early stopping ----
        if val_loss < history.best_val_loss - 1e-6:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                if log is not None:
                    log.info(
                        "Early stopping at epoch %d (best epoch=%d, best_val=%.6f)",
                        epoch, history.best_epoch, history.best_val_loss,
                    )
                break

    if best_state is None:
        # No improvement at all — keep final weights.
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state, history

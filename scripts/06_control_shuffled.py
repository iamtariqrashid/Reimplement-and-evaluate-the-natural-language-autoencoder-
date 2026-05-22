"""Step 6 (optional) — control experiment: shuffled explanations.

Trains the same MLP architecture on the same split, but with the
explanation-embedding rows shuffled relative to the activations. The gap
between the real and shuffled FVE measures how much of the reconstruction
quality actually comes from the verbalization (vs the activation prior).

Usage:
    python scripts/06_control_shuffled.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import summarize
from src.reconstructor import MLPReconstructor, train_reconstructor
from src.utils import (
    ensure_dirs,
    get_device,
    get_logger,
    load_config,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shuffled-explanations control.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ensure_dirs(cfg)
    log = get_logger()

    data_dir = Path(cfg["paths"]["data_dir"])
    res_dir = Path(cfg["paths"]["results_dir"])

    # Load the same artifacts step 04/05 used
    acts = np.load(data_dir / cfg["activations"]["file"]).astype(np.float32)
    expl_emb = np.load(data_dir / "explanation_embeddings.npy").astype(np.float32)
    splits = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
    train_idx = np.array(splits["train"], dtype=int)
    val_idx = np.array(splits["val"], dtype=int)
    test_idx = np.array(splits["test"], dtype=int)
    log.info(
        "Loaded acts=%s | expl_emb=%s | train/val/test=%d/%d/%d",
        acts.shape, expl_emb.shape, len(train_idx), len(val_idx), len(test_idx),
    )

    # ---- Build a shuffled view of explanation embeddings ----
    rng = np.random.default_rng(cfg["seed"] + 1)
    perm = np.arange(acts.shape[0])
    rng.shuffle(perm)
    n_fixed = int((perm == np.arange(acts.shape[0])).sum())
    log.info("Shuffle permutation built (%d rows accidentally unchanged)", n_fixed)
    expl_emb_shuf = expl_emb[perm]

    device = get_device()
    r = cfg["reconstructor"]
    X = torch.from_numpy(expl_emb_shuf)
    y = torch.from_numpy(acts)

    # ---- Train identical MLP on shuffled pairs ----
    model = MLPReconstructor(
        input_dim=expl_emb.shape[1],
        output_dim=acts.shape[1],
        hidden_dims=tuple(r["hidden_dims"]),
        dropout=r["dropout"],
    )
    log.info("Training control MLP (shuffled explanations)...")
    best_state, history = train_reconstructor(
        model,
        X[train_idx], y[train_idx], X[val_idx], y[val_idx],
        lr=r["lr"],
        weight_decay=r["weight_decay"],
        batch_size=r["batch_size"],
        epochs=r["epochs"],
        early_stopping_patience=r["early_stopping_patience"],
        device=device,
        log=log,
    )

    # ---- Evaluate on the test split ----
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        y_pred = model(X[test_idx].to(device)).cpu().numpy()
    y_true = acts[test_idx]
    summary = summarize(y_true, y_pred)
    summary.pop("_per_sample", None)

    log.info("=== Control (shuffled) test results ===")
    for k, v in summary.items():
        log.info("  %-18s = %.6f", k, v)

    # ---- Compare to real run if available ----
    real_path = res_dir / cfg["eval"]["results_file"]
    comparison = {"control_shuffled": summary}
    if real_path.exists():
        real = json.loads(real_path.read_text(encoding="utf-8"))
        comparison["real"] = real
        comparison["delta_fve_global"] = real["fve_global"] - summary["fve_global"]
        comparison["delta_fve_sample_mean"] = (
            real["fve_sample_mean"] - summary["fve_sample_mean"]
        )
        log.info(
            "real FVE=%.4f | shuffled FVE=%.4f | delta=%.4f (real minus shuffled)",
            real["fve_global"], summary["fve_global"],
            comparison["delta_fve_global"],
        )

    out_path = res_dir / "control_shuffled.json"
    out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    log.info("Saved control results to %s", out_path)


if __name__ == "__main__":
    main()

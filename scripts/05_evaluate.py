"""Step 5 — evaluate the trained reconstructor on the held-out test split.

Usage:
    python scripts/05_evaluate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import summarize
from src.plotting import hist_plot, loss_curve, per_cluster_bar
from src.reconstructor import MLPReconstructor
from src.utils import (
    ensure_dirs,
    get_device,
    get_logger,
    load_config,
    read_jsonl,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstructor.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ensure_dirs(cfg)
    log = get_logger()

    data_dir = Path(cfg["paths"]["data_dir"])
    res_dir = Path(cfg["paths"]["results_dir"])
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])

    # ---- Load everything ----
    acts = np.load(data_dir / cfg["activations"]["file"]).astype(np.float32)
    expl_emb = np.load(data_dir / "explanation_embeddings.npy").astype(np.float32)
    verbs = read_jsonl(data_dir / cfg["verbalizer"]["verbalizations_file"])
    splits = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
    test_idx = np.array(splits["test"], dtype=int)
    log.info("Test set size: %d", len(test_idx))

    ckpt_path = ckpt_dir / cfg["reconstructor"]["checkpoint_file"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MLPReconstructor(
        input_dim=ckpt["input_dim"],
        output_dim=ckpt["output_dim"],
        hidden_dims=tuple(ckpt["hidden_dims"]),
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    device = get_device()
    model.to(device)

    # ---- Predict on test split ----
    with torch.no_grad():
        X_test = torch.from_numpy(expl_emb[test_idx]).to(device)
        y_pred = model(X_test).cpu().numpy()
    y_true = acts[test_idx]

    summary = summarize(y_true, y_pred)
    per_sample = summary.pop("_per_sample")

    log.info("=== Test results (held out) ===")
    for k, v in summary.items():
        log.info("  %-18s = %.6f", k, v)

    # ---- Save scalar results ----
    results_path = res_dir / cfg["eval"]["results_file"]
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved scalar results to %s", results_path)

    # ---- Per-cluster breakdown ----
    test_clusters = np.array([verbs[i]["cluster"] for i in test_idx])
    cluster_rows = []
    for c in sorted(set(test_clusters.tolist())):
        mask = test_clusters == c
        cluster_rows.append({
            "cluster": int(c),
            "n_test_samples": int(mask.sum()),
            "fve_mean": float(per_sample["fve"][mask].mean()),
            "cosine_mean": float(per_sample["cosine"][mask].mean()),
            "mse_mean": float(per_sample["mse"][mask].mean()),
        })
    per_cluster_path = res_dir / cfg["eval"]["per_cluster_file"]
    pd.DataFrame(cluster_rows).to_csv(per_cluster_path, index=False)
    log.info("Saved per-cluster results to %s", per_cluster_path)

    # ---- Qualitative examples (best, worst, and representative) ----
    num_qual = cfg["eval"]["num_qualitative"]
    sorted_idx = np.argsort(-per_sample["fve"])  # high FVE first
    n_each = max(1, num_qual // 3)
    pick = np.concatenate([
        sorted_idx[:n_each],                # best
        sorted_idx[-n_each:],               # worst
        sorted_idx[len(sorted_idx) // 2 - n_each // 2 :
                   len(sorted_idx) // 2 + n_each // 2 + 1],  # median
    ])
    seen, qual_rows = set(), []
    for j in pick:
        if j in seen:
            continue
        seen.add(j)
        global_i = int(test_idx[j])
        rec = verbs[global_i]
        qual_rows.append({
            "test_position": int(j),
            "global_id": global_i,
            "cluster": int(rec["cluster"]),
            "explanation": rec["explanation"],
            "text": rec["text"],
            "fve": float(per_sample["fve"][j]),
            "cosine": float(per_sample["cosine"][j]),
            "mse": float(per_sample["mse"][j]),
        })
    qual_path = res_dir / cfg["eval"]["qualitative_file"]
    pd.DataFrame(qual_rows).to_csv(qual_path, index=False)
    log.info("Saved %d qualitative examples to %s", len(qual_rows), qual_path)

    # ---- Plots ----
    hist_plot(
        per_sample["fve"],
        title="Per-sample FVE (test split)",
        xlabel="FVE",
        out_path=res_dir / "fve_histogram.png",
    )
    hist_plot(
        per_sample["cosine"],
        title="Per-sample cosine similarity (test split)",
        xlabel="cosine similarity",
        out_path=res_dir / "cosine_histogram.png",
    )
    per_cluster_bar(
        test_clusters,
        per_sample["fve"],
        metric_name="FVE",
        out_path=res_dir / "fve_per_cluster.png",
    )
    log.info("Saved plots to %s", res_dir)

    # ---- Loss curve (if training history was saved) ----
    hist_file = res_dir / "training_history.json"
    if hist_file.exists():
        hist = json.loads(hist_file.read_text(encoding="utf-8"))
        if hist.get("train_loss"):
            loss_curve(
                hist["train_loss"], hist["val_loss"],
                out_path=res_dir / "training_loss.png",
            )
            log.info("Saved loss curve")


if __name__ == "__main__":
    main()

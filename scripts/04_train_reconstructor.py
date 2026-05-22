"""Step 4 — embed explanations and train the MLP reconstructor.

Usage:
    python scripts/04_train_reconstructor.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reconstructor import (
    MLPReconstructor,
    make_splits,
    train_reconstructor,
)
from src.utils import (
    ensure_dirs,
    get_device,
    get_logger,
    load_config,
    read_jsonl,
    set_seed,
)


def embed_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: torch.device,
    log,
) -> np.ndarray:
    """Encode a list of strings to a (N, embed_dim) float32 array."""
    from sentence_transformers import SentenceTransformer  # lazy import

    log.info("Loading sentence embedder: %s", model_name)
    enc = SentenceTransformer(model_name, device=str(device))
    embs = enc.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embs.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP reconstructor.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ensure_dirs(cfg)
    log = get_logger()

    data_dir = Path(cfg["paths"]["data_dir"])
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])

    acts = np.load(data_dir / cfg["activations"]["file"])
    verbs = read_jsonl(data_dir / cfg["verbalizer"]["verbalizations_file"])
    if len(verbs) != acts.shape[0]:
        raise ValueError(
            f"verbalizations ({len(verbs)}) != activations ({acts.shape[0]}); "
            "re-run scripts 02 and 03 together."
        )
    explanations = [r["explanation"] for r in verbs]

    device = get_device()
    log.info("Device: %s", device)

    # ---- Embed explanations ----
    expl_emb = embed_texts(
        explanations,
        model_name=cfg["embedder"]["name"],
        batch_size=cfg["embedder"]["batch_size"],
        device=device,
        log=log,
    )
    log.info("Explanation embeddings: %s | activations: %s", expl_emb.shape, acts.shape)

    # ---- Splits (saved so evaluation uses the same partition) ----
    r = cfg["reconstructor"]
    splits = make_splits(
        n=acts.shape[0],
        val_frac=r["val_frac"],
        test_frac=r["test_frac"],
        seed=cfg["seed"],
    )
    splits_path = data_dir / "splits.json"
    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(
            {k: v.tolist() for k, v in asdict(splits).items()},
            f,
        )
    log.info(
        "Splits: train=%d val=%d test=%d (saved to %s)",
        len(splits.train), len(splits.val), len(splits.test), splits_path,
    )

    X = torch.from_numpy(expl_emb)
    y = torch.from_numpy(acts.astype(np.float32))
    X_train, y_train = X[splits.train], y[splits.train]
    X_val, y_val = X[splits.val], y[splits.val]

    # ---- Model ----
    model = MLPReconstructor(
        input_dim=expl_emb.shape[1],
        output_dim=acts.shape[1],
        hidden_dims=tuple(r["hidden_dims"]),
        dropout=r["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "Reconstructor MLP: in=%d hidden=%s out=%d | %d params",
        expl_emb.shape[1], r["hidden_dims"], acts.shape[1], n_params,
    )

    # ---- Train ----
    best_state, history = train_reconstructor(
        model,
        X_train, y_train, X_val, y_val,
        lr=r["lr"],
        weight_decay=r["weight_decay"],
        batch_size=r["batch_size"],
        epochs=r["epochs"],
        early_stopping_patience=r["early_stopping_patience"],
        device=device,
        log=log,
    )

    # ---- Save ----
    ckpt_path = ckpt_dir / r["checkpoint_file"]
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": expl_emb.shape[1],
            "output_dim": acts.shape[1],
            "hidden_dims": list(r["hidden_dims"]),
            "dropout": r["dropout"],
            "embedder_name": cfg["embedder"]["name"],
        },
        ckpt_path,
    )
    log.info("Saved best checkpoint to %s (best epoch=%d, val_mse=%.6f)",
             ckpt_path, history.best_epoch, history.best_val_loss)

    # Save training history alongside results for plotting / record-keeping
    hist_path = Path(cfg["paths"]["results_dir"]) / "training_history.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(history), f, indent=2)
    log.info("Saved training history to %s", hist_path)

    # Also cache the explanation embeddings so step 05 can reuse them
    emb_path = data_dir / "explanation_embeddings.npy"
    np.save(emb_path, expl_emb)
    log.info("Cached explanation embeddings at %s", emb_path)


if __name__ == "__main__":
    main()

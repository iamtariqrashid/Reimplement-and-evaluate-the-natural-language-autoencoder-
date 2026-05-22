"""Step 2 — extract hidden activations from the small LLM.

Usage:
    python scripts/02_collect_activations.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.activations import extract_activations
from src.model_utils import load_model
from src.utils import (
    ensure_dirs,
    get_device,
    get_logger,
    load_config,
    read_jsonl,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LLM activations.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ensure_dirs(cfg)
    log = get_logger()

    data_dir = Path(cfg["paths"]["data_dir"])
    texts_path = data_dir / cfg["data"]["texts_file"]
    if not texts_path.exists():
        raise FileNotFoundError(
            f"{texts_path} missing — run scripts/01_prepare_data.py first."
        )
    records = read_jsonl(texts_path)
    texts = [r["text"] for r in records]
    log.info("Loaded %d texts", len(texts))

    device = get_device()
    log.info("Device: %s", device)

    m = cfg["model"]
    loaded = load_model(
        name=m["name"],
        layer_index=m["layer_index"],
        dtype=m["dtype"],
        device=device,
    )
    log.info(
        "Model %s loaded | layers=%d | hidden=%d | extracting layer=%d | pooling=%s",
        m["name"],
        loaded.num_hidden_layers,
        loaded.hidden_size,
        loaded.layer_index,
        m["pooling"],
    )

    acts = extract_activations(
        loaded,
        texts,
        max_length=m["max_length"],
        batch_size=m["batch_size"],
        pooling=m["pooling"],
    )
    log.info("Activations shape: %s | dtype=%s", acts.shape, acts.dtype)

    out_path = data_dir / cfg["activations"]["file"]
    np.save(out_path, acts)
    log.info("Saved activations to %s", out_path)

    meta = {
        "model_name": m["name"],
        "num_hidden_layers": loaded.num_hidden_layers,
        "extracted_layer": loaded.layer_index,
        "pooling": m["pooling"],
        "hidden_size": int(loaded.hidden_size),
        "num_samples": int(acts.shape[0]),
        "max_length": m["max_length"],
        "dtype": m["dtype"],
    }
    meta_path = out_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved metadata to %s", meta_path)


if __name__ == "__main__":
    main()

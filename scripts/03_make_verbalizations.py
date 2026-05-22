"""Step 3 — cluster activations and verbalize each cluster.

Usage:
    python scripts/03_make_verbalizations.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import (
    ensure_dirs,
    get_logger,
    load_config,
    read_jsonl,
    set_seed,
    write_jsonl,
)
from src.verbalizer import build_cluster_explanations


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cluster verbalizations.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ensure_dirs(cfg)
    log = get_logger()

    data_dir = Path(cfg["paths"]["data_dir"])
    texts_path = data_dir / cfg["data"]["texts_file"]
    acts_path = data_dir / cfg["activations"]["file"]
    for p in (texts_path, acts_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run earlier scripts first.")

    records = read_jsonl(texts_path)
    texts = [r["text"] for r in records]
    activations = np.load(acts_path)

    if activations.shape[0] != len(texts):
        raise ValueError(
            f"Activation rows ({activations.shape[0]}) != texts ({len(texts)})."
        )

    v = cfg["verbalizer"]
    log.info(
        "Running KMeans (k=%d, n_init=%d) on %s activations...",
        v["num_clusters"], v["kmeans_n_init"], activations.shape,
    )
    labels, infos = build_cluster_explanations(
        activations=activations,
        texts=texts,
        num_clusters=v["num_clusters"],
        n_init=v["kmeans_n_init"],
        top_k_examples=v["top_k_examples"],
        seed=cfg["seed"],
    )
    log.info("Built explanations for %d clusters", len(infos))
    log.info("Sample explanation (cluster 0): %s", infos[0].explanation)

    # 1) clusters.json — human-readable cluster card
    clusters_path = data_dir / v["clusters_file"]
    with clusters_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(ci) for ci in infos], f, indent=2, ensure_ascii=False)
    log.info("Saved cluster info to %s", clusters_path)

    # 2) verbalizations.jsonl — per-sample assignment, fuels the reconstructor
    explanation_by_cluster = {ci.cluster_id: ci.explanation for ci in infos}
    out_records = [
        {
            "id": r["id"],
            "text": r["text"],
            "cluster": int(labels[i]),
            "explanation": explanation_by_cluster[int(labels[i])],
        }
        for i, r in enumerate(records)
    ]
    verb_path = data_dir / v["verbalizations_file"]
    write_jsonl(verb_path, out_records)
    log.info("Saved per-sample verbalizations to %s", verb_path)


if __name__ == "__main__":
    main()

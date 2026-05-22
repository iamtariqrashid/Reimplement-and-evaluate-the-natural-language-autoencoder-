"""Step 3b (preferred) — LLM-based natural-language verbalizer.

Replaces the TF-IDF templated verbalizer with the small LM itself: we prompt
SmolLM2 (the same model whose activations we extract) to summarize each
cluster's representative texts in one short sentence. The output replaces
data/verbalizations.jsonl in the format used by scripts 04/05/06, so the
rest of the pipeline does not need to change.

Two modes:
  --mode cluster  (default)  : one LLM call per KMeans cluster (fast, ~K calls)
  --mode per-sample          : one LLM call per fine-grained neighbourhood
                               (slow; uses smaller K' from --fine-k)

Usage:
    python scripts/03b_llm_verbalize.py
    python scripts/03b_llm_verbalize.py --mode per-sample --fine-k 64
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_verbalizer import (
    verbalize_clusters_with_llm,
    verbalize_per_sample_with_llm,
)
from src.model_utils import load_model
from src.utils import (
    ensure_dirs,
    get_device,
    get_logger,
    load_config,
    read_jsonl,
    set_seed,
    write_jsonl,
)
from src.verbalizer import fit_kmeans


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-based verbalizer.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode", choices=["cluster", "per-sample"], default="cluster",
        help="cluster: one call per KMeans cluster (fast). "
             "per-sample: one call per fine-grained neighbourhood (slow).",
    )
    parser.add_argument(
        "--fine-k", type=int, default=64,
        help="Per-sample mode only: number of cache buckets (rough granularity).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=60,
        help="Generation budget per call.",
    )
    parser.add_argument(
        "--neighbours", type=int, default=None,
        help="Number of neighbour texts to show the LM (default: verbalizer.top_k_examples).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ensure_dirs(cfg)
    log = get_logger()

    data_dir = Path(cfg["paths"]["data_dir"])

    # ---- Load data ----
    records = read_jsonl(data_dir / cfg["data"]["texts_file"])
    texts = [r["text"] for r in records]
    activations = np.load(data_dir / cfg["activations"]["file"])
    if activations.shape[0] != len(texts):
        raise ValueError(
            f"texts ({len(texts)}) and activations ({activations.shape[0]}) "
            "must have the same length."
        )

    v = cfg["verbalizer"]
    neighbours = args.neighbours if args.neighbours is not None else v["top_k_examples"]
    log.info(
        "LLM verbalizer mode=%s | neighbours=%d | max_new_tokens=%d",
        args.mode, neighbours, args.max_new_tokens,
    )

    # ---- Load LLM (re-used for generation) ----
    device = get_device()
    m = cfg["model"]
    loaded = load_model(
        name=m["name"], layer_index=m["layer_index"], dtype=m["dtype"], device=device,
    )
    log.info("Loaded %s for generation on %s", m["name"], device)

    if args.mode == "cluster":
        # Use the same KMeans grouping as the original script for an
        # apples-to-apples comparison, then replace each cluster's TF-IDF
        # template with an LM-generated sentence.
        log.info("Running KMeans (k=%d) on %s...", v["num_clusters"], activations.shape)
        km, labels = fit_kmeans(
            activations,
            num_clusters=v["num_clusters"],
            n_init=v["kmeans_n_init"],
            seed=cfg["seed"],
        )
        infos, exp_by_cluster = verbalize_clusters_with_llm(
            loaded=loaded,
            activations=activations,
            texts=texts,
            cluster_labels=labels,
            centroids=km.cluster_centers_,
            top_k_examples=neighbours,
            max_new_tokens=args.max_new_tokens,
        )
        # Persist cluster cards
        clusters_path = data_dir / "clusters_llm.json"
        with clusters_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(ci) for ci in infos], f, indent=2, ensure_ascii=False)
        log.info("Saved LLM cluster cards to %s", clusters_path)

        # Save the fitted KMeans model so the Streamlit app and notebook
        # can load the exact same cluster assignments without refitting.
        import joblib
        km_path = data_dir / "kmeans.joblib"
        joblib.dump(km, km_path)
        log.info("Saved KMeans model to %s", km_path)

        out_records = [
            {
                "id": records[i]["id"],
                "text": records[i]["text"],
                "cluster": int(labels[i]),
                "explanation": exp_by_cluster[int(labels[i])],
            }
            for i in range(len(records))
        ]

    else:  # per-sample
        log.info(
            "Per-sample LLM verbalization with cache_by=%d-cluster bucket", args.fine_k,
        )
        km_fine, labels_fine = fit_kmeans(
            activations,
            num_clusters=args.fine_k,
            n_init=v["kmeans_n_init"],
            seed=cfg["seed"],
        )

        def cache_key(i: int) -> int:
            return int(labels_fine[i])

        explanations = verbalize_per_sample_with_llm(
            loaded=loaded,
            activations=activations,
            texts=texts,
            top_k_neighbours=neighbours,
            max_new_tokens=args.max_new_tokens,
            cache_by=cache_key,
        )

        out_records = [
            {
                "id": records[i]["id"],
                "text": records[i]["text"],
                "cluster": int(labels_fine[i]),
                "explanation": explanations[i],
            }
            for i in range(len(records))
        ]

    verb_path = data_dir / v["verbalizations_file"]
    write_jsonl(verb_path, out_records)
    log.info(
        "Wrote %d records to %s (overwriting any previous verbalization)",
        len(out_records), verb_path,
    )

    # Log a few examples so the user can sanity-check the LM output without
    # opening the JSONL by hand.
    log.info("--- Sample explanations ---")
    seen: set[str] = set()
    for r in out_records:
        if r["explanation"] in seen:
            continue
        seen.add(r["explanation"])
        log.info("c=%d | %s", r["cluster"], r["explanation"][:160])
        if len(seen) >= 6:
            break


if __name__ == "__main__":
    main()

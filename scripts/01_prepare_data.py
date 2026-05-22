"""Step 1 — prepare and save the text dataset.

Usage:
    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --config config.yaml --num-samples 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly: python scripts/01_prepare_data.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_text_samples
from src.utils import ensure_dirs, get_logger, load_config, set_seed, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare text dataset.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override config.data.num_samples",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ensure_dirs(cfg)
    log = get_logger()

    d = cfg["data"]
    num_samples = args.num_samples if args.num_samples is not None else d["num_samples"]

    log.info(
        "Loading %s/%s split=%s, target=%d samples",
        d["dataset_name"], d["dataset_config"], d["split"], num_samples,
    )

    records = load_text_samples(
        dataset_name=d["dataset_name"],
        dataset_config=d["dataset_config"],
        split=d["split"],
        num_samples=num_samples,
        min_chars=d["min_chars"],
        max_chars=d["max_chars"],
        seed=cfg["seed"],
    )

    out_path = Path(cfg["paths"]["data_dir"]) / d["texts_file"]
    write_jsonl(out_path, records)
    log.info("Wrote %d records to %s", len(records), out_path)

    # Quick sanity preview
    log.info("Example record: %s", {**records[0], "text": records[0]["text"][:120] + "..."})


if __name__ == "__main__":
    main()

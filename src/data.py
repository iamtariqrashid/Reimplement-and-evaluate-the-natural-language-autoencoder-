"""Dataset loading and text filtering.

Dataset requirement: always load Salesforce/wikitext with wikitext-2-raw-v1.
Do not silently switch to another dataset source.
"""
from __future__ import annotations

import re
import random
from typing import Any

from datasets import load_dataset

# WikiText-2 section headings look like " = Title = " or " = = Sub = = "
_HEADING_RE = re.compile(r"^\s*=+\s*.+\s*=+\s*$")


def load_text_samples(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    num_samples: int,
    min_chars: int,
    max_chars: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Load WikiText from Hugging Face, filter by length and quality, sample uniformly.

    Expected call from config:
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    Returns a list of records: {"id": int, "text": str}.
    """
    if dataset_name != "Salesforce/wikitext":
        raise ValueError(
            f"Expected dataset_name='Salesforce/wikitext', got '{dataset_name}'. "
            "Update config.yaml — do not silently switch to another dataset."
        )
    if dataset_config != "wikitext-2-raw-v1":
        raise ValueError(
            f"Expected dataset_config='wikitext-2-raw-v1', got '{dataset_config}'. "
            "Update config.yaml — do not silently switch to another config."
        )

    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split=split,
        trust_remote_code=False,
    )

    # Filter, deduplicate, and remove WikiText section headings / fragments
    seen: set[str] = set()
    candidates: list[str] = []
    for ex in ds:
        t = (ex["text"] or "").strip()
        if not t:
            continue
        if _HEADING_RE.match(t):
            continue
        if not (min_chars <= len(t) <= max_chars):
            continue
        if t in seen:
            continue
        seen.add(t)
        candidates.append(t)

    if len(candidates) < num_samples:
        raise ValueError(
            f"Only {len(candidates)} candidates after filtering, "
            f"but num_samples={num_samples}. Lower num_samples or loosen filters."
        )

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:num_samples]

    return [{"id": i, "text": t} for i, t in enumerate(selected)]

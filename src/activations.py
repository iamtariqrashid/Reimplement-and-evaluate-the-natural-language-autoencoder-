"""Extract hidden-state activations from a HF causal LM."""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from .model_utils import LoadedModel


Pooling = Literal["mean", "last"]


@torch.no_grad()
def extract_activations(
    loaded: LoadedModel,
    texts: list[str],
    max_length: int,
    batch_size: int,
    pooling: Pooling = "mean",
) -> np.ndarray:
    """Run texts through the LM and return pooled hidden states.

    Returns
    -------
    np.ndarray of shape (len(texts), hidden_size), dtype float32.
    """
    model = loaded.model
    tok = loaded.tokenizer
    layer = loaded.layer_index
    device = loaded.device

    all_vecs: list[np.ndarray] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="extract"):
        batch_texts = texts[start : start + batch_size]
        enc = tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        outputs = model(**enc, output_hidden_states=True)

        # hidden_states is a tuple of length (num_layers + 1):
        #   [0] = token embeddings, [i] = output of layer i for i >= 1.
        hs = outputs.hidden_states[layer]            # (B, T, H)
        attn = enc["attention_mask"]                 # (B, T)

        if pooling == "mean":
            mask = attn.unsqueeze(-1).to(hs.dtype)   # (B, T, 1)
            summed = (hs * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts                 # (B, H)
        elif pooling == "last":
            # index of the last non-pad token per row
            last_idx = attn.sum(dim=1) - 1           # (B,)
            pooled = hs[torch.arange(hs.size(0), device=device), last_idx]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        all_vecs.append(pooled.float().cpu().numpy())

    return np.concatenate(all_vecs, axis=0)

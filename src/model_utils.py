"""Load the small LLM + tokenizer with hidden-state output enabled."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import torch_dtype


@dataclass
class LoadedModel:
    model: Any                  # PreTrainedModel
    tokenizer: Any              # PreTrainedTokenizer
    hidden_size: int            # activation dimension
    num_hidden_layers: int      # transformer depth
    layer_index: int            # which layer we will extract from
    device: torch.device


def _resolve_layer_index(num_layers: int, requested: int) -> int:
    """Auto-select 2/3-depth layer when requested == -1, else clamp."""
    if requested == -1:
        return max(1, int(round((2 / 3) * num_layers)))
    if requested < 0 or requested > num_layers:
        raise ValueError(
            f"layer_index {requested} out of range for model with {num_layers} layers."
        )
    return requested


def load_model(
    name: str,
    layer_index: int,
    dtype: str,
    device: torch.device,
) -> LoadedModel:
    """Load tokenizer + causal LM and configure pad token + dtype.

    Resulting model has output_hidden_states=True and is in eval mode.
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        # Many decoder-only models ship without a pad token; reuse EOS.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch_dtype(dtype),
        output_hidden_states=True,
    )
    model.to(device)
    model.eval()

    # Make sure pad id is wired through model config too.
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # `num_hidden_layers` is the standard HF attribute; fall back if missing.
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = len(model.model.layers) if hasattr(model, "model") else 12
    resolved_layer = _resolve_layer_index(num_layers, layer_index)

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = model.config.n_embd  # GPT-2 naming fallback

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        layer_index=resolved_layer,
        device=device,
    )

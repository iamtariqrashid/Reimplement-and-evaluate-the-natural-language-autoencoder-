"""Shared utilities: seeding, config loading, IO helpers, device, logging."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


# ---------- Reproducibility ----------
def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Hash seed for any Python set/dict iteration sensitivity
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------- Config ----------
def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the YAML config file as a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path.resolve()}. "
            "Run scripts from the project root."
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create data/results/checkpoints directories if missing."""
    for key in ("data_dir", "results_dir", "checkpoints_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


# ---------- JSONL ----------
def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------- Device ----------
def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def torch_dtype(name: str) -> torch.dtype:
    name = name.lower()
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }.get(name, torch.float32)


# ---------- Logging ----------
def get_logger(name: str = "nla") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

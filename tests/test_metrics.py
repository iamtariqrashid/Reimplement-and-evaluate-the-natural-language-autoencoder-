"""Tests for src/metrics.py.

These pin down the FVE formula against known-answer cases. FVE is the
primary metric of the task; if our implementation drifts from the
standard definition, every reported number is meaningless.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import (
    cosine_per_sample,
    fve_global,
    fve_per_sample,
    mse_per_sample,
    summarize,
)


# --------------------------------------------------------------------------- #
# FVE                                                                          #
# --------------------------------------------------------------------------- #
def test_fve_global_perfect_reconstruction_is_one():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(50, 8))
    assert fve_global(y, y.copy()) == pytest.approx(1.0)


def test_fve_global_predicting_the_mean_is_zero():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(50, 8))
    y_pred = np.broadcast_to(y.mean(axis=0, keepdims=True), y.shape).copy()
    assert fve_global(y, y_pred) == pytest.approx(0.0, abs=1e-12)


def test_fve_global_worse_than_mean_is_negative():
    """A predictor doing worse than 'just predict the mean' should give FVE<0."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=(50, 4))
    # Predict the opposite sign — strictly worse than predicting the mean.
    y_pred = -y
    assert fve_global(y, y_pred) < 0


def test_fve_per_sample_average_equals_global_when_uniform_norm():
    """When per-sample SS_tot is constant, the global FVE equals mean of per-sample."""
    # Construct y so that each row has the same squared distance from the mean.
    y = np.array(
        [[ 1.0,  0.0],
         [-1.0,  0.0],
         [ 0.0,  1.0],
         [ 0.0, -1.0]],
        dtype=np.float64,
    )
    # Some non-trivial prediction
    y_pred = y * 0.5
    g = fve_global(y, y_pred)
    p = fve_per_sample(y, y_pred).mean()
    assert g == pytest.approx(p, rel=1e-10)


def test_fve_matches_sklearn_r2_when_single_dim():
    """In 1-D, FVE equals scikit-learn's R^2 score."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    y = rng.normal(size=(100, 1))
    y_pred = y + rng.normal(scale=0.3, size=y.shape)
    assert fve_global(y, y_pred) == pytest.approx(
        sklearn_metrics.r2_score(y.ravel(), y_pred.ravel()), rel=1e-8,
    )


def test_fve_zero_variance_inputs_return_nan():
    y = np.zeros((10, 4), dtype=np.float64)
    out = fve_global(y, y + 0.01)
    assert np.isnan(out)


# --------------------------------------------------------------------------- #
# MSE / cosine                                                                 #
# --------------------------------------------------------------------------- #
def test_mse_per_sample_known_values():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 2.0], [4.0, 4.0]])
    # row 0: 0; row 1: ((4-3)^2 + 0)/2 = 0.5
    assert np.allclose(mse_per_sample(y, y_pred), [0.0, 0.5])


def test_cosine_per_sample_identity_is_one():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(30, 5))
    assert np.allclose(cosine_per_sample(y, y), np.ones(30), atol=1e-6)


def test_cosine_per_sample_orthogonal_is_zero():
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(cosine_per_sample(y, y_pred), [0.0, 0.0])


def test_cosine_per_sample_opposite_is_minus_one():
    y = np.array([[1.0, 2.0, 3.0]])
    y_pred = -y
    assert cosine_per_sample(y, y_pred)[0] == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# summarize()                                                                  #
# --------------------------------------------------------------------------- #
def test_summarize_returns_expected_keys():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(40, 6))
    y_pred = y + rng.normal(scale=0.1, size=y.shape)
    s = summarize(y, y_pred)
    for k in (
        "fve_global", "mse_mean", "mse_median",
        "cosine_mean", "cosine_median",
        "fve_sample_mean", "fve_sample_median",
        "_per_sample",
    ):
        assert k in s, f"missing key {k}"
    assert s["_per_sample"]["mse"].shape == (40,)
    assert s["_per_sample"]["cosine"].shape == (40,)
    assert s["_per_sample"]["fve"].shape == (40,)


def test_summarize_perfect_reconstruction():
    rng = np.random.default_rng(4)
    y = rng.normal(size=(20, 3))
    s = summarize(y, y.copy())
    assert s["fve_global"] == pytest.approx(1.0)
    assert s["mse_mean"] == pytest.approx(0.0)
    assert s["cosine_mean"] == pytest.approx(1.0, abs=1e-6)

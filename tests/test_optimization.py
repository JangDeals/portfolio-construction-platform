"""
Tests for src/optimization/mean_variance.py.
Focuses on verifying constraints are actually respected, not just that
optimization runs without error.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pytest
import numpy as np
import pandas as pd
from src.optimization.mean_variance import minimum_variance_portfolio


@pytest.fixture
def sample_cov_matrix():
    """A small, simple 3-asset covariance matrix for testing."""
    return pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.01]],
        columns=["A", "B", "C"], index=["A", "B", "C"]
    )


def test_weights_sum_to_one(sample_cov_matrix):
    weights = minimum_variance_portfolio(sample_cov_matrix)
    assert weights.sum() == pytest.approx(1.0, abs=1e-4)


def test_weights_respect_max_weight_constraint(sample_cov_matrix):
    max_w = 0.5
    weights = minimum_variance_portfolio(sample_cov_matrix, max_weight=max_w)
    assert (weights <= max_w + 1e-4).all()


def test_weights_are_non_negative_long_only(sample_cov_matrix):
    """
    Confirms the long-only constraint: no asset should have a negative
    weight (no short-selling).
    """
    weights = minimum_variance_portfolio(sample_cov_matrix)
    assert (weights >= -1e-6).all()  # tiny tolerance for solver noise near zero
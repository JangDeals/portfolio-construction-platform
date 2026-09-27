"""
Tests for src/risk/risk_engine.py.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pytest
import pandas as pd
import numpy as np
from src.risk.risk_engine import maximum_drawdown, herfindahl_index, historical_var


def test_maximum_drawdown_known_case():
    """
    Wealth path 100 -> 110 -> 105 -> 90 -> 95 -> 120 -> 80, hand-traced in
    Module 9: max drawdown is -33.33% from the peak of 120 (day 6) to 80
    (day 7), NOT from the earlier local peak of 110.
    """
    values = [100, 110, 105, 90, 95, 120, 80]
    cumulative_returns = pd.Series([v / values[0] - 1 for v in values],
                                     index=pd.date_range("2023-01-01", periods=7))

    result = maximum_drawdown(cumulative_returns)
    expected_dd = (80 - 120) / 120  # -0.3333...

    assert result["max_drawdown"] == pytest.approx(expected_dd, abs=1e-4)
    # Confirm it correctly identifies the peak BEFORE the trough, not the
    # earlier local peak of 110 on day 2
    assert pd.Timestamp(result["peak_date"]) == cumulative_returns.index[5]
    assert pd.Timestamp(result["trough_date"]) == cumulative_returns.index[6]


def test_herfindahl_index_equal_weight():
    """
    HHI of an equally-weighted n-asset portfolio must equal exactly 1/n.
    """
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    result = herfindahl_index(weights)
    assert result == pytest.approx(1 / 4, abs=1e-6)


def test_herfindahl_index_full_concentration():
    """
    A 100%-single-asset portfolio must have HHI = 1.0 exactly, the maximum
    possible value.
    """
    weights = np.array([1.0, 0.0, 0.0])
    result = herfindahl_index(weights)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_historical_var_matches_percentile_by_hand():
    """
    10 returns, sorted. The 90% VaR should equal the 10th percentile
    (the value below which only 10% of returns fall), which for exactly
    10 sorted values is the smallest one.
    """
    returns = pd.Series([-0.10, -0.05, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
    result = historical_var(returns, confidence_level=0.90)

    expected = -np.percentile(returns, 10)
    assert result == pytest.approx(expected, abs=1e-6)
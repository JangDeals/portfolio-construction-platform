# tests/test_edge_cases.py
"""
Edge case tests: deliberately probing boundary and failure conditions
across multiple modules, per Module 25's requirements.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pytest
import pandas as pd
import numpy as np
from src.portfolio.construction import custom_allocation_portfolio
from src.features.return_calculator import calculate_simple_returns
from src.risk.risk_engine import maximum_drawdown


def test_custom_allocation_rejects_weights_not_summing_to_one():
    """
    Weights summing to 0.9 (not 1.0) must raise ValueError, not silently
    proceed with an invalid portfolio.
    """
    bad_weights = {"A": 0.5, "B": 0.4}  # sums to 0.9
    with pytest.raises(ValueError):
        custom_allocation_portfolio(bad_weights)


def test_custom_allocation_accepts_floating_point_near_one():
    """
    Weights summing to 0.9999999999 (a harmless floating-point rounding
    artifact) should NOT raise an error -- this tests the np.isclose
    tolerance decision from Module 11.
    """
    near_one_weights = {"A": 0.333333333, "B": 0.333333333, "C": 0.333333334}
    result = custom_allocation_portfolio(near_one_weights)
    assert result.sum() == pytest.approx(1.0, abs=1e-6)


def test_simple_returns_on_empty_dataframe():
    """
    An empty price DataFrame should return an empty result, not crash.
    """
    empty_prices = pd.DataFrame(columns=["A", "B"])
    result = calculate_simple_returns(empty_prices)
    assert len(result) == 0


def test_simple_returns_with_single_row():
    """
    A single row of prices has no prior period to compute a return from --
    result should be empty, not raise an error or return garbage.
    """
    single_row = pd.DataFrame({"A": [100]}, index=pd.date_range("2023-01-01", periods=1))
    result = calculate_simple_returns(single_row)
    assert len(result) == 0


def test_maximum_drawdown_with_only_gains():
    """
    A portfolio that only ever went up should have max drawdown of exactly
    0% -- there was never a decline from any peak.
    """
    cumulative_returns = pd.Series([0.0, 0.05, 0.10, 0.15],
                                     index=pd.date_range("2023-01-01", periods=4))
    result = maximum_drawdown(cumulative_returns)
    assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-6)
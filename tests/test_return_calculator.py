"""
Tests for src/features/return_calculator.py.
Uses small, hand-computable examples so expected results can be verified
independently, rather than testing against the full real dataset.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pytest
import pandas as pd
import numpy as np
from src.features.return_calculator import (
    calculate_simple_returns, calculate_log_returns,
    calculate_cumulative_returns, calculate_annualized_return
)


def test_simple_returns_basic():
    """
    Prices 100 -> 110 -> 99 should give simple returns of +10% and -10%.
    """
    prices = pd.DataFrame({"A": [100, 110, 99]}, index=pd.date_range("2023-01-01", periods=3))
    returns = calculate_simple_returns(prices)

    assert len(returns) == 2  # first row dropped, no prior price to compare
    assert returns["A"].iloc[0] == pytest.approx(0.10, abs=1e-6)
    assert returns["A"].iloc[1] == pytest.approx(-0.10, abs=1e-6)


def test_log_returns_basic():
    """
    Log return of 100 -> 110 should equal ln(1.10), a known, hand-computable value.
    """
    prices = pd.DataFrame({"A": [100, 110]}, index=pd.date_range("2023-01-01", periods=2))
    log_returns = calculate_log_returns(prices)

    expected = np.log(1.10)
    assert log_returns["A"].iloc[0] == pytest.approx(expected, abs=1e-6)


def test_cumulative_returns_compounding():
    """
    Simple returns of +5%, -5% should compound to -0.25%, NOT 0% (the naive
    sum). This directly tests the exact compounding lesson from Module 6.
    """
    simple_returns = pd.DataFrame({"A": [0.05, -0.05]}, index=pd.date_range("2023-01-01", periods=2))
    cumulative = calculate_cumulative_returns(simple_returns)

    expected_final = (1.05 * 0.95) - 1  # = -0.0025
    assert cumulative["A"].iloc[-1] == pytest.approx(expected_final, abs=1e-6)
    assert cumulative["A"].iloc[-1] != pytest.approx(0.0, abs=1e-6)  # explicitly NOT naive sum


def test_annualized_return_known_case():
    """
    A constant 1% daily return for 252 days should annualize to
    (1.01)^(252/252) - 1 = 1% exactly, since n_periods == trading_days_per_year.
    """
    simple_returns = pd.DataFrame({"A": [0.01] * 252}, index=pd.date_range("2023-01-01", periods=252))
    annualized = calculate_annualized_return(simple_returns, trading_days_per_year=252)

    total_return = 1.01 ** 252 - 1
    expected = (1 + total_return) ** (252 / 252) - 1
    assert annualized["A"] == pytest.approx(expected, abs=1e-6)
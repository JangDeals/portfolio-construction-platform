"""
Tests for src/portfolio/portfolio_theory.py.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pytest
import numpy as np
import pandas as pd
from src.portfolio.portfolio_theory import portfolio_expected_return, portfolio_variance, sharpe_ratio


def test_portfolio_expected_return_two_assets():
    """
    50/50 portfolio of assets returning 10% and 20% should have expected
    return of exactly 15%.
    """
    weights = np.array([0.5, 0.5])
    expected_returns = pd.Series([0.10, 0.20])

    result = portfolio_expected_return(weights, expected_returns)
    assert result == pytest.approx(0.15, abs=1e-6)


def test_portfolio_variance_matches_two_asset_formula():
    """
    Verify the matrix formula w.T . Sigma . w matches the manual two-asset
    formula: w1^2*var1 + w2^2*var2 + 2*w1*w2*cov12.
    """
    weights = np.array([0.6, 0.4])
    cov_matrix = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]])  # var1=0.04, var2=0.09, cov=0.01

    result = portfolio_variance(weights, cov_matrix)
    expected = (0.6**2 * 0.04) + (0.4**2 * 0.09) + (2 * 0.6 * 0.4 * 0.01)

    assert result == pytest.approx(expected, abs=1e-6)


def test_diversification_reduces_variance_below_naive_average():
    """
    With negative correlation, portfolio variance should be strictly less
    than a naive weighted average of individual variances -- this is the
    core mathematical proof from Module 8.
    """
    weights = np.array([0.5, 0.5])
    var1, var2 = 0.04, 0.04
    cov_matrix = pd.DataFrame([[var1, -0.02], [-0.02, var2]])  # negative covariance

    port_var = portfolio_variance(weights, cov_matrix)
    naive_avg_var = 0.5 * var1 + 0.5 * var2

    assert port_var < naive_avg_var


def test_sharpe_ratio_known_case():
    """
    Return 10%, risk-free 2%, volatility 8% -> Sharpe = 0.08/0.08 = 1.0.
    """
    result = sharpe_ratio(portfolio_return=0.10, portfolio_vol=0.08, risk_free_rate=0.02)
    assert result == pytest.approx(1.0, abs=1e-6)
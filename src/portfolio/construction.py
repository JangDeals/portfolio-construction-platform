"""
Portfolio construction module.
Implements multiple portfolio construction strategies: equal weight,
custom allocation, and inverse volatility (risk-based) weighting.
"""

import numpy as np
import pandas as pd


def equal_weight_portfolio(tickers: list[str]) -> pd.Series:
    """
    Construct an equal-weight portfolio.

    Parameters
    ----------
    tickers : list[str]

    Returns
    -------
    pd.Series
        Weights indexed by ticker, summing to 1.
    """
    n = len(tickers)
    weights = pd.Series([1 / n] * n, index=tickers)
    return weights


def custom_allocation_portfolio(weights_dict: dict) -> pd.Series:
    """
    Construct a portfolio from manually specified weights.

    Parameters
    ----------
    weights_dict : dict
        Ticker -> weight mapping. Must sum to 1 (validated).

    Returns
    -------
    pd.Series
        Weights indexed by ticker.

    Raises
    ------
    ValueError
        If weights do not sum to 1 (within floating point tolerance).
    """
    weights = pd.Series(weights_dict)

    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1.0, got {weights.sum():.6f}")

    return weights


def inverse_volatility_portfolio(returns: pd.DataFrame, trading_days_per_year: int = 252) -> pd.Series:
    """
    Construct a risk-based portfolio using inverse volatility weighting.

    w_i = (1/sigma_i) / sum_j(1/sigma_j)

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns, tickers as columns.
    trading_days_per_year : int, default 252

    Returns
    -------
    pd.Series
        Weights indexed by ticker, summing to 1.
    """
    annualized_vol = returns.std() * np.sqrt(trading_days_per_year)
    inverse_vol = 1 / annualized_vol
    weights = inverse_vol / inverse_vol.sum()
    return weights


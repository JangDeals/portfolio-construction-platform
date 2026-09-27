"""
Factor analysis module.
Provides a simplified, honest factor exposure analysis covering Market,
Momentum, and Volatility factors, using data already computed elsewhere in
this project. Value, Quality, and Size factors are explicitly out of scope
(see module docstring notes) since they require external fundamental data
not available for our ETF universe.
"""

import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from src.risk.risk_engine import beta as calculate_beta


def market_factor_exposure(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Market factor exposure, measured as beta vs. a broad market benchmark
    (SPY). This is the same calculation as Module 9's beta function,
    reframed explicitly as a factor exposure measure.

    Parameters
    ----------
    portfolio_returns : pd.Series
    benchmark_returns : pd.Series

    Returns
    -------
    float
        Market factor exposure (beta).
    """
    return calculate_beta(portfolio_returns, benchmark_returns)


def calculate_momentum_scores(prices: pd.DataFrame, lookback_days: int = 252) -> pd.Series:
    """
    Calculate trailing momentum for each asset as its total return over the
    lookback period (default: trailing 12 months / 252 trading days),
    measured as of the most recent date in prices.

    Parameters
    ----------
    prices : pd.DataFrame
    lookback_days : int, default 252

    Returns
    -------
    pd.Series
        Trailing return over the lookback window, per ticker.
    """
    recent_prices = prices.iloc[-1]
    lookback_prices = prices.iloc[-lookback_days]
    momentum = (recent_prices / lookback_prices) - 1
    return momentum


def portfolio_momentum_exposure(weights: pd.Series, momentum_scores: pd.Series) -> float:
    """
    Calculate a portfolio's weighted-average momentum exposure: the
    weighted sum of each held asset's trailing momentum score.

    Parameters
    ----------
    weights : pd.Series
    momentum_scores : pd.Series

    Returns
    -------
    float
        Weighted average trailing momentum of the portfolio's holdings.
    """
    return (weights * momentum_scores).sum()


def portfolio_volatility_factor_exposure(weights: pd.Series, individual_volatilities: pd.Series) -> float:
    """
    Calculate a portfolio's weighted-average exposure to the volatility
    factor: the weighted sum of each held asset's own standalone
    volatility. A lower value indicates a stronger tilt toward the
    "low volatility" side of this factor.

    Parameters
    ----------
    weights : pd.Series
    individual_volatilities : pd.Series
        Each asset's own annualized volatility.

    Returns
    -------
    float
        Weighted average standalone volatility of the portfolio's holdings.
    """
    return (weights * individual_volatilities).sum()



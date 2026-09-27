"""
Performance analytics module.
Implements CAGR, Sortino ratio, Calmar ratio, and Information ratio.
"""

import numpy as np
import pandas as pd


def calculate_cagr(cumulative_returns: pd.Series) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR) from a cumulative returns
    series.

    CAGR = (V_end / V_start) ^ (1 / n_years) - 1

    Parameters
    ----------
    cumulative_returns : pd.Series
        Cumulative return series with a DatetimeIndex.

    Returns
    -------
    float
        CAGR.
    """
    wealth_index = 1 + cumulative_returns
    v_start = wealth_index.iloc[0]
    v_end = wealth_index.iloc[-1]

    n_days = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days
    n_years = n_days / 365.25

    return (v_end / v_start) ** (1 / n_years) - 1

def sortino_ratio(portfolio_return: float, downside_vol: float, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sortino ratio: excess return per unit of downside risk.

    Sortino = (E(Rp) - Rf) / downside_volatility

    Parameters
    ----------
    portfolio_return : float
        Annualized portfolio return.
    downside_vol : float
        Annualized downside volatility (from src.risk.risk_engine).
    risk_free_rate : float, default 0.0

    Returns
    -------
    float
        Sortino ratio.
    """
    return (portfolio_return - risk_free_rate) / downside_vol


def calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """
    Calculate the Calmar ratio: annualized return relative to maximum
    drawdown severity.

    Calmar = Annualized Return / |Maximum Drawdown|

    Parameters
    ----------
    annualized_return : float
    max_drawdown : float
        Maximum drawdown (expected as a negative number, e.g. -0.3387).

    Returns
    -------
    float
        Calmar ratio.
    """
    return annualized_return / abs(max_drawdown)


def information_ratio(portfolio_return: float, benchmark_return: float, tracking_err: float) -> float:
    """
    Calculate the Information ratio: excess return over a benchmark per
    unit of tracking error.

    IR = (E(Rp) - E(Rb)) / Tracking Error

    Parameters
    ----------
    portfolio_return : float
        Annualized portfolio return.
    benchmark_return : float
        Annualized benchmark return.
    tracking_err : float
        Annualized tracking error (from src.risk.risk_engine).

    Returns
    -------
    float
        Information ratio.
    """
    return (portfolio_return - benchmark_return) / tracking_err


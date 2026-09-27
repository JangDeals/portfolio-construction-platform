"""
Portfolio risk engine.
Implements downside volatility, maximum drawdown, VaR (historical and
parametric), CVaR, tracking error, beta, and concentration risk (HHI).
"""

import numpy as np
import pandas as pd
from scipy import stats


def downside_volatility(returns: pd.Series, trading_days_per_year: int = 252) -> float:
    """
    Calculate annualized downside volatility (semi-deviation), using only
    returns below zero.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns for a single asset or portfolio.
    trading_days_per_year : int, default 252

    Returns
    -------
    float
        Annualized downside volatility.
    """
    downside_returns = returns[returns < 0]
    daily_downside_vol = np.sqrt((downside_returns ** 2).sum() / len(returns))
    return daily_downside_vol * np.sqrt(trading_days_per_year)


def maximum_drawdown(cumulative_returns: pd.Series) -> dict:
    """
    Calculate maximum drawdown from a cumulative returns series.

    Parameters
    ----------
    cumulative_returns : pd.Series
        Cumulative return series (e.g. output of calculate_cumulative_returns
        for a single asset or portfolio).

    Returns
    -------
    dict
        max_drawdown (float, negative), peak_date, trough_date.
    """
    wealth_index = 1 + cumulative_returns  # convert back to a growth-of-$1 series
    running_peak = wealth_index.cummax()
    drawdown = (wealth_index - running_peak) / running_peak

    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    peak_date = wealth_index[:trough_date].idxmax()

    return {
        "max_drawdown": max_dd,
        "peak_date": str(peak_date.date()),
        "trough_date": str(trough_date.date()),
    }

def historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Historical VaR: the empirical percentile of the historical
    return distribution.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.
    confidence_level : float, default 0.95
        e.g. 0.95 for 95% VaR.

    Returns
    -------
    float
        VaR as a positive number representing the loss threshold.
    """
    percentile = (1 - confidence_level) * 100
    var = np.percentile(returns, percentile)
    return -var  # express as a positive loss number


def parametric_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Parametric VaR, assuming returns are normally distributed.

    VaR = -(mu + z * sigma)

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.
    confidence_level : float, default 0.95

    Returns
    -------
    float
        VaR as a positive number representing the loss threshold.
    """
    mu = returns.mean()
    sigma = returns.std()
    z_score = stats.norm.ppf(1 - confidence_level)  # e.g. -1.645 for 95%
    var = -(mu + z_score * sigma)
    return var

def historical_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Historical CVaR (Expected Shortfall): the average of all
    returns worse than the Historical VaR threshold.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.
    confidence_level : float, default 0.95

    Returns
    -------
    float
        CVaR as a positive number.
    """
    var_threshold = -historical_var(returns, confidence_level)  # back to a raw return cutoff
    tail_returns = returns[returns <= var_threshold]
    return -tail_returns.mean()

def tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    """
    Calculate annualized tracking error: the volatility of the difference
    between portfolio and benchmark returns.

    Parameters
    ----------
    portfolio_returns : pd.Series
    benchmark_returns : pd.Series
    trading_days_per_year : int, default 252

    Returns
    -------
    float
        Annualized tracking error.
    """
    active_returns = portfolio_returns - benchmark_returns
    return active_returns.std() * np.sqrt(trading_days_per_year)


def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate portfolio beta relative to a benchmark.

    Beta = Cov(Rp, Rb) / Var(Rb)

    Parameters
    ----------
    portfolio_returns : pd.Series
    benchmark_returns : pd.Series

    Returns
    -------
    float
        Beta.
    """
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns, ddof=1)
    return covariance / benchmark_variance

def herfindahl_index(weights: np.ndarray) -> float:
    """
    Calculate the Herfindahl-Hirschman Index (HHI) as a measure of portfolio
    concentration.

    HHI = sum(w_i^2)

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights, must sum to 1.

    Returns
    -------
    float
        HHI. Ranges from 1/n (perfectly equal-weighted, most diversified)
        to 1 (fully concentrated in a single asset).
    """
    return np.sum(weights ** 2)


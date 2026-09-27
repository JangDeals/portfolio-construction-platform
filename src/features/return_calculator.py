"""
Return calculation module.
Computes simple returns, log returns, cumulative returns, and annualized
returns from price data.
"""

import pandas as pd
import numpy as np

# --- Return Calculation Functions ---
def calculate_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate period-over-period simple (arithmetic) returns.

    R_t = (P_t - P_{t-1}) / P_{t-1}

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with dates as index, tickers as columns.

    Returns
    -------
    pd.DataFrame
        Simple returns, same shape as input minus the first row (which has
        no prior price to compute a return from).
    """
    simple_returns = prices.pct_change().dropna(how="all")
    return simple_returns

# --- Log Return Calculation Function ---
def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate period-over-period log (continuously compounded) returns.

    r_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with dates as index, tickers as columns.

    Returns
    -------
    pd.DataFrame
        Log returns, same shape as input minus the first row.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    return log_returns

# --- Cumulative Return Calculation Function ---
def calculate_cumulative_returns(simple_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative return from a series of simple returns.

    Cumulative Return_t = product(1 + R_i) - 1, for i = 1 to t

    Parameters
    ----------
    simple_returns : pd.DataFrame
        Simple returns with dates as index, tickers as columns.

    Returns
    -------
    pd.DataFrame
        Cumulative return at each point in time, starting from 0 at t=0.
    """
    cumulative_returns = (1 + simple_returns).cumprod() - 1
    return cumulative_returns

# --- Annualized Return Calculation Function ---
def calculate_annualized_return(simple_returns: pd.DataFrame, trading_days_per_year: int = 252) -> pd.Series:
    """
    Calculate the annualized return over the full period covered by
    simple_returns.

    Annualized Return = (1 + Total Return) ^ (trading_days_per_year / n) - 1

    Parameters
    ----------
    simple_returns : pd.DataFrame
        Simple returns with dates as index, tickers as columns.
    trading_days_per_year : int, default 252
        Market-standard convention for trading days in a year.

    Returns
    -------
    pd.Series
        Annualized return for each ticker.
    """
    total_return = (1 + simple_returns).prod() - 1
    n_periods = len(simple_returns)
    annualized_return = (1 + total_return) ** (trading_days_per_year / n_periods) - 1
    return annualized_return


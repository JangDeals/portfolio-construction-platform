"""
Portfolio rebalancing engine.
Tracks how a portfolio's actual weights drift from target over time as
asset prices move, and generates calendar-based or threshold-based
rebalancing recommendations.
"""

import numpy as np
import pandas as pd


def compute_drifted_weights(target_weights: pd.Series, prices: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """
    Simulate how a portfolio's actual weights drift away from target over
    time, assuming target_weights were established on start_date and never
    rebalanced afterward.

    Parameters
    ----------
    target_weights : pd.Series
        Target weights indexed by ticker, established at start_date.
    prices : pd.DataFrame
        Price data with dates as index, tickers as columns.
    start_date : str
        The date the portfolio was formed at target_weights.

    Returns
    -------
    pd.DataFrame
        Actual (drifted) weights over time, dates as index, tickers as
        columns.
    """
    prices_from_start = prices[prices.index >= start_date]
    initial_prices = prices_from_start.iloc[0]

    # Assume $1 total invested at start_date, split according to target_weights.
    # Track how many "units" of each asset that buys, then value those units
    # forward through time using actual price changes.
    units_held = target_weights / initial_prices
    dollar_values_over_time = prices_from_start * units_held

    total_value_over_time = dollar_values_over_time.sum(axis=1)
    drifted_weights = dollar_values_over_time.div(total_value_over_time, axis=0)

    return drifted_weights


def calendar_rebalance_dates(prices: pd.DataFrame, start_date: str, frequency: str = "QS") -> list:
    """
    Generate a list of calendar-based rebalancing dates.

    Parameters
    ----------
    prices : pd.DataFrame
    start_date : str
    frequency : str, default "QS"
        Pandas frequency string: "MS" (monthly), "QS" (quarterly),
        "YS" (annually).

    Returns
    -------
    list
        List of rebalancing dates (as Timestamps) within the price data's
        date range.
    """
    end_date = prices.index.max()
    return list(pd.date_range(start=start_date, end=end_date, freq=frequency))[1:]  # skip start_date itself


def detect_threshold_breaches(drifted_weights: pd.DataFrame, target_weights: pd.Series, threshold: float = 0.05) -> pd.DataFrame:
    """
    Identify every date on which any asset's drift from target exceeds the
    given threshold.

    Parameters
    ----------
    drifted_weights : pd.DataFrame
        Output of compute_drifted_weights.
    target_weights : pd.Series
    threshold : float, default 0.05
        Maximum allowed absolute deviation from target (e.g. 0.05 = 5
        percentage points) before a rebalance is triggered.

    Returns
    -------
    pd.DataFrame
        One row per date where a breach occurred, with columns: date,
        ticker, target_weight, actual_weight, drift.
    """
    drift = drifted_weights.subtract(target_weights, axis=1)
    breaches = drift.abs() > threshold

    breach_records = []
    for date in drifted_weights.index[breaches.any(axis=1)]:
        for ticker in drifted_weights.columns:
            if breaches.loc[date, ticker]:
                breach_records.append({
                    "date": date,
                    "ticker": ticker,
                    "target_weight": target_weights[ticker],
                    "actual_weight": drifted_weights.loc[date, ticker],
                    "drift": drift.loc[date, ticker],
                })

    return pd.DataFrame(breach_records)


def compute_rebalancing_trades(current_weights: pd.Series, target_weights: pd.Series) -> dict:
    """
    Compute the trades required to bring a portfolio back to target weights,
    and the resulting turnover.

    Parameters
    ----------
    current_weights : pd.Series
        Actual (drifted) weights at the moment of rebalancing.
    target_weights : pd.Series

    Returns
    -------
    dict
        trades (pd.Series, positive = buy, negative = sell) and turnover
        (float, total trading required as a fraction of portfolio value).
    """
    trades = target_weights - current_weights
    turnover = trades.abs().sum() / 2  # each trade has a buy and sell side; avoid double-counting

    return {"trades": trades, "turnover": turnover}




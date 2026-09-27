"""
Stress testing module.
Evaluates portfolio performance under historical crisis periods and
hypothetical shock scenarios. Historical and hypothetical results are
kept explicitly separate and never blended.
"""

import numpy as np
import pandas as pd


HISTORICAL_SCENARIOS = {
    "2008 Global Financial Crisis": ("2008-09-01", "2009-03-31"),
    "2020 COVID Crash": ("2020-02-01", "2020-03-31"),
    "2022 Rate-Hike Bear Market": ("2022-01-01", "2022-10-31"),
}


def run_historical_scenario(weights: pd.Series, simple_returns: pd.DataFrame, start_date: str, end_date: str) -> dict:
    """
    Evaluate a fixed-weight portfolio's actual historical performance
    during a specific real date range.

    Parameters
    ----------
    weights : pd.Series
    simple_returns : pd.DataFrame
    start_date, end_date : str

    Returns
    -------
    dict
        total_return, max_drawdown, volatility, and per-asset contribution
        to total return over the window.
    """
    window_returns = simple_returns[(simple_returns.index >= start_date) & (simple_returns.index <= end_date)]

    portfolio_returns = window_returns.dot(weights.values)
    cumulative = (1 + portfolio_returns).cumprod() - 1
    total_return = cumulative.iloc[-1]

    wealth_index = 1 + cumulative
    running_peak = wealth_index.cummax()
    max_dd = ((wealth_index - running_peak) / running_peak).min()

    volatility = portfolio_returns.std() * np.sqrt(252)

    # Per-asset contribution: each asset's weight times its own cumulative
    # return over the window, approximating its share of total portfolio return.
    asset_cumulative = (1 + window_returns).prod() - 1
    contribution = weights * asset_cumulative

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "volatility": volatility,
        "asset_contribution": contribution,
    }
    
    
HYPOTHETICAL_SCENARIOS = {
    "Equity Crash": {"SPY": -0.30, "QQQ": -0.30, "VEA": -0.30, "VWO": -0.30},
    "Inflation/Rate Shock": {"IEF": -0.15, "TLT": -0.15, "SHY": -0.15, "GLD": 0.10, 
                              "SPY": -0.05, "QQQ": -0.05, "VEA": -0.05, "VWO": -0.05},
    "Simultaneous Equity & Bond Crash": {"SPY": -0.25, "QQQ": -0.25, "VEA": -0.25, "VWO": -0.25,
                                           "IEF": -0.10, "TLT": -0.10, "SHY": -0.10, "GLD": 0.05},
}


def run_hypothetical_scenario(weights: pd.Series, shock: dict, tickers: list[str]) -> dict:
    """
    Evaluate a portfolio's instantaneous return under an assumed one-time
    shock applied to specific assets. Assets not named in the shock dict
    are assumed unchanged (0% return).

    Parameters
    ----------
    weights : pd.Series
    shock : dict
        Ticker -> assumed instantaneous return.
    tickers : list[str]
        Full universe, to fill in 0% for any asset not named in the shock.

    Returns
    -------
    dict
        portfolio_return and asset_contribution.
    """
    shock_returns = pd.Series({t: shock.get(t, 0.0) for t in tickers})
    portfolio_return = weights @ shock_returns
    contribution = weights * shock_returns

    return {"portfolio_return": portfolio_return, "asset_contribution": contribution}




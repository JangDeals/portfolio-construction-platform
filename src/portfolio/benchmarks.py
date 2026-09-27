"""
Benchmark construction module.
Builds standard reference portfolios (60/40, equal-weight multi-asset) used
to evaluate whether optimized/constructed portfolios add genuine value over
simple, low-effort alternatives.
"""

import numpy as np
import pandas as pd


def sixty_forty_benchmark(tickers: list[str], equity_ticker: str = "SPY", bond_ticker: str = "IEF") -> pd.Series:
    """
    Construct the classic 60/40 benchmark: 60% equity, 40% bonds.
    All other tickers in the universe receive zero weight.

    Parameters
    ----------
    tickers : list[str]
        Full asset universe (for consistent indexing with other strategies).
    equity_ticker : str, default "SPY"
    bond_ticker : str, default "IEF"

    Returns
    -------
    pd.Series
        Weights indexed by ticker, summing to 1.
    """
    weights = pd.Series(0.0, index=tickers)
    weights[equity_ticker] = 0.60
    weights[bond_ticker] = 0.40
    return weights


def equal_weight_multiasset_benchmark(tickers: list[str]) -> pd.Series:
    """
    Construct the equal-weight multi-asset benchmark — identical in
    construction to Module 8's equal-weight portfolio, but used here in a
    distinct role: as a naive reference point against which optimized
    strategies should demonstrate genuine added value.

    Parameters
    ----------
    tickers : list[str]

    Returns
    -------
    pd.Series
        Weights indexed by ticker, summing to 1.
    """
    n = len(tickers)
    return pd.Series([1 / n] * n, index=tickers)
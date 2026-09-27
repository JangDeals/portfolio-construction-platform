"""
Portfolio backtesting module.
Implements train/test split and walk-forward backtesting with rebalancing
and transaction cost modeling.
"""

import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from src.optimization.mean_variance import minimum_variance_portfolio, maximum_sharpe_portfolio
from src.portfolio.construction import equal_weight_portfolio


def compute_inputs(returns: pd.DataFrame, trading_days_per_year: int = 252) -> tuple:
    """
    Compute annualized expected returns and covariance matrix from a
    returns window. Used to generate strategy weights using ONLY data
    available up to that point.
    """
    annualized_returns = (1 + returns).prod() ** (trading_days_per_year / len(returns)) - 1
    cov_matrix = returns.cov() * trading_days_per_year
    return annualized_returns, cov_matrix


def get_strategy_weights(strategy_name: str, train_window: pd.DataFrame, risk_free_rate: float = 0.02) -> pd.Series:
    """
    Compute portfolio weights for a given strategy, using ONLY the supplied
    training window's data — never data from outside this window.
    """
    exp_returns, cov = compute_inputs(train_window)
    tickers = train_window.columns.tolist()

    if strategy_name == "Equal Weight":
        return equal_weight_portfolio(tickers)
    elif strategy_name == "Minimum Variance":
        return minimum_variance_portfolio(cov)
    elif strategy_name == "Maximum Sharpe":
        return maximum_sharpe_portfolio(exp_returns, cov, risk_free_rate)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

def backtest_single_split(
    strategy_name: str,
    train_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    transaction_cost: float = 0.001,
) -> dict:
    """
    Run a single train/test backtest: compute weights using train_returns
    only, then apply those FIXED weights to test_returns to simulate
    out-of-sample performance.

    Parameters
    ----------
    strategy_name : str
    train_returns, test_returns : pd.DataFrame
    transaction_cost : float, default 0.001
        One-time cost applied at portfolio formation, as a fraction of
        capital, to reflect the cost of establishing these positions
        (e.g. 0.001 = 10 basis points).

    Returns
    -------
    dict
        weights, test_period_returns (pd.Series), and cumulative_return
        after accounting for the one-time formation transaction cost.
    """
    weights = get_strategy_weights(strategy_name, train_returns)

    # Portfolio's daily returns over the OUT-OF-SAMPLE test period,
    # using weights that were fixed based on train data alone.
    test_portfolio_returns = test_returns.dot(weights.values)

    # Apply a one-time transaction cost at formation (turnover from cash
    # into these positions), modeled as a reduction to day-1 return.
    test_portfolio_returns.iloc[0] -= transaction_cost

    cumulative_return = (1 + test_portfolio_returns).prod() - 1

    return {
        "weights": weights,
        "test_returns": test_portfolio_returns,
        "cumulative_return": cumulative_return,
    }
    

def walk_forward_backtest(
    strategy_name: str,
    returns: pd.DataFrame,
    initial_train_years: int = 10,
    transaction_cost: float = 0.001,
    risk_free_rate: float = 0.02,
) -> dict:
    """
    Run a walk-forward backtest: repeatedly re-estimate weights using an
    expanding training window, then apply those weights to the following
    year's returns, stitching together a continuous out-of-sample return
    series across the full test span.

    Parameters
    ----------
    strategy_name : str
    returns : pd.DataFrame
        Full daily simple returns dataset.
    initial_train_years : int, default 10
        Number of years of data required before the first rebalance.
    transaction_cost : float, default 0.001
        Applied at each rebalancing date, as a fraction of capital.
    risk_free_rate : float, default 0.02

    Returns
    -------
    dict
        oos_returns (pd.Series, the full stitched-together out-of-sample
        daily return series), rebalance_dates, and weights_history (dict
        of date -> weights, for inspection).
    """
    start_date = returns.index.min()
    end_date = returns.index.max()

    first_rebalance = start_date + pd.DateOffset(years=initial_train_years)
    rebalance_dates = pd.date_range(start=first_rebalance, end=end_date, freq="YS")

    all_oos_returns = []
    weights_history = {}

    for i, rebal_date in enumerate(rebalance_dates):
        train_window = returns[returns.index < rebal_date]

        next_rebal_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else end_date + pd.Timedelta(days=1)
        test_window = returns[(returns.index >= rebal_date) & (returns.index < next_rebal_date)]

        if len(test_window) == 0:
            continue

        weights = get_strategy_weights(strategy_name, train_window, risk_free_rate)
        weights_history[str(rebal_date.date())] = weights

        period_returns = test_window.dot(weights.values)
        period_returns.iloc[0] -= transaction_cost  # cost of rebalancing into new weights

        all_oos_returns.append(period_returns)

    oos_returns = pd.concat(all_oos_returns)

    return {
        "oos_returns": oos_returns,
        "rebalance_dates": rebalance_dates,
        "weights_history": weights_history,
    }
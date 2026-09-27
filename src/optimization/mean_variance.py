"""
Mean-variance optimization module.
Implements Minimum Variance Portfolio, Maximum Sharpe Portfolio, and
Efficient Frontier construction using scipy.optimize.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Internal helper: portfolio variance for a given weight vector."""
    return weights.T @ cov_matrix @ weights


def _negative_sharpe(weights: np.ndarray, expected_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float) -> float:
    """
    Internal helper: negative Sharpe ratio for a given weight vector.
    Negative because scipy.optimize.minimize only minimizes — maximizing
    Sharpe is equivalent to minimizing its negative.
    """
    port_return = weights @ expected_returns
    port_vol = np.sqrt(_portfolio_variance(weights, cov_matrix))
    return -(port_return - risk_free_rate) / port_vol


def minimum_variance_portfolio(
    cov_matrix: pd.DataFrame,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
    asset_class_map: dict = None,
    asset_class_bounds: dict = None,
    max_hhi: float = None,
) -> pd.Series:
    """
    Solve for the Minimum Variance Portfolio, with optional constraints.

    Parameters
    ----------
    cov_matrix : pd.DataFrame
    max_weight : float, default 1.0
        Maximum weight allowed for any single asset (e.g. 0.3 for 30%).
    min_weight : float, default 0.0
        Minimum weight allowed for any asset that is held above zero is
        not enforced here (that requires integer/binary variables beyond
        SLSQP's scope) — this sets the floor for every asset uniformly.
    asset_class_map, asset_class_bounds, max_hhi : optional
        See _build_constraints.

    Returns
    -------
    pd.Series
        Optimal weights indexed by ticker.
    """
    n = len(cov_matrix)
    tickers = cov_matrix.columns

    initial_guess = np.array([1 / n] * n)
    bounds = tuple((min_weight, max_weight) for _ in range(n))
    constraints = _build_constraints(
        n, asset_class_map=asset_class_map, asset_class_bounds=asset_class_bounds, max_hhi=max_hhi
    )

    result = minimize(
        _portfolio_variance,
        initial_guess,
        args=(cov_matrix.values,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return pd.Series(result.x, index=tickers)

def maximum_sharpe_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
    asset_class_map: dict = None,
    asset_class_bounds: dict = None,
    max_hhi: float = None,
) -> pd.Series:
    """
    Solve for the Maximum Sharpe Ratio Portfolio, with optional constraints.
    See minimum_variance_portfolio for parameter details.
    """
    n = len(cov_matrix)
    tickers = cov_matrix.columns

    initial_guess = np.array([1 / n] * n)
    bounds = tuple((min_weight, max_weight) for _ in range(n))
    constraints = _build_constraints(
        n, asset_class_map=asset_class_map, asset_class_bounds=asset_class_bounds, max_hhi=max_hhi
    )

    result = minimize(
        _negative_sharpe,
        initial_guess,
        args=(expected_returns.values, cov_matrix.values, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return pd.Series(result.x, index=tickers)

def minimum_variance_for_target_return(expected_returns: pd.Series, cov_matrix: pd.DataFrame, target_return: float) -> pd.Series:
    """
    Solve for the minimum-variance portfolio subject to achieving at least
    a specified target return. Used to trace out the Efficient Frontier.

    Parameters
    ----------
    expected_returns : pd.Series
    Annualized expected return for each asset.
    cov_matrix : pd.DataFrame
    target_return : float

    Returns
    -------
    pd.Series
        Optimal weights indexed by ticker.
    """
    n = len(cov_matrix)
    tickers = cov_matrix.columns

    initial_guess = np.array([1 / n] * n)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: w @ expected_returns.values - target_return},
    )

    result = minimize(
        _portfolio_variance,
        initial_guess,
        args=(cov_matrix.values,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        return None  # some target returns may be infeasible; handled by caller

    return pd.Series(result.x, index=tickers)


def build_efficient_frontier(expected_returns: pd.Series, cov_matrix: pd.DataFrame, n_points: int = 50) -> pd.DataFrame:
    """
    Construct the Efficient Frontier by solving the minimum-variance problem
    across a range of target returns.

    Parameters
    ----------
    expected_returns : pd.Series
    cov_matrix : pd.DataFrame
    n_points : int, default 50
        Number of points to trace along the frontier.

    Returns
    -------
    pd.DataFrame
        Columns: target_return, volatility, and one column per asset weight.
    """
    min_return = expected_returns.min()
    max_return = expected_returns.max()
    target_returns = np.linspace(min_return, max_return, n_points)

    frontier_points = []
    for target in target_returns:
        weights = minimum_variance_for_target_return(expected_returns, cov_matrix, target)
        if weights is None:
            continue  # skip infeasible target returns

        vol = np.sqrt(_portfolio_variance(weights.values, cov_matrix.values))
        point = {"target_return": target, "volatility": vol}
        point.update(weights.to_dict())
        frontier_points.append(point)

    return pd.DataFrame(frontier_points)

def _build_constraints(
    n: int,
    expected_returns: np.ndarray = None,
    asset_class_map: dict = None,
    asset_class_bounds: dict = None,
    max_hhi: float = None,
    target_volatility: float = None,
    cov_matrix: np.ndarray = None,
) -> list:
    """
    Build the list of SciPy constraint dictionaries shared across
    optimization functions.

    Parameters
    ----------
    n : int
        Number of assets.
    expected_returns : np.ndarray, optional
        Required only if target_return-style constraints are used elsewhere.
    asset_class_map : dict, optional
        Ticker -> asset class name (e.g. {"SPY": "Equity", "IEF": "FixedIncome"}).
    asset_class_bounds : dict, optional
        Asset class name -> (min_weight, max_weight) tuple, e.g.
        {"Equity": (0.4, 0.7), "FixedIncome": (0.2, 0.4)}.
    max_hhi : float, optional
        Maximum allowed Herfindahl-Hirschman Index (concentration cap).
    target_volatility : float, optional
        If provided, constrains portfolio volatility to equal this value
        exactly. Requires cov_matrix.
    cov_matrix : np.ndarray, optional
        Required only if target_volatility is used.

    Returns
    -------
    list
        List of SciPy constraint dictionaries. Always includes the
        weights-sum-to-1 constraint.
    """
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if asset_class_map is not None and asset_class_bounds is not None:
        tickers = list(asset_class_map.keys())
        for asset_class, (min_w, max_w) in asset_class_bounds.items():
            class_indices = [i for i, t in enumerate(tickers) if asset_class_map[t] == asset_class]

            constraints.append({
                "type": "ineq",
                "fun": lambda w, idx=class_indices, lo=min_w: np.sum(w[idx]) - lo
            })
            constraints.append({
                "type": "ineq",
                "fun": lambda w, idx=class_indices, hi=max_w: hi - np.sum(w[idx])
            })

    if max_hhi is not None:
        constraints.append({
            "type": "ineq",
            "fun": lambda w, cap=max_hhi: cap - np.sum(w ** 2)
        })

    if target_volatility is not None and cov_matrix is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda w, cov=cov_matrix, target=target_volatility: np.sqrt(w.T @ cov @ w) - target
        })

    return constraints

def target_volatility_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    target_volatility: float,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> pd.Series:
    """
    Solve for the portfolio that maximizes expected return subject to
    hitting a specific target volatility exactly.

    Parameters
    ----------
    expected_returns : pd.Series
    cov_matrix : pd.DataFrame
    target_volatility : float
        Desired annualized portfolio volatility.
    max_weight, min_weight : float

    Returns
    -------
    pd.Series
        Optimal weights indexed by ticker.
    """
    n = len(cov_matrix)
    tickers = cov_matrix.columns

    initial_guess = np.array([1 / n] * n)
    bounds = tuple((min_weight, max_weight) for _ in range(n))
    constraints = _build_constraints(
        n, target_volatility=target_volatility, cov_matrix=cov_matrix.values
    )

    def _negative_return(weights, exp_ret):
        return -(weights @ exp_ret)

    result = minimize(
        _negative_return,
        initial_guess,
        args=(expected_returns.values,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return pd.Series(result.x, index=tickers)


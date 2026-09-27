"""
Advanced portfolio optimization module.
Implements Risk Parity and Maximum Diversification portfolio construction.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _risk_contributions(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate each asset's risk contribution to total portfolio volatility.

    RC_i = w_i * (Sigma . w)_i / sigma_p
    """
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    marginal_contributions = cov_matrix @ weights
    risk_contributions = weights * marginal_contributions / port_vol
    return risk_contributions


def _risk_parity_objective(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Objective function for Risk Parity: sum of squared differences between
    each pair of assets' risk contributions. Minimized to zero when all
    risk contributions are exactly equal.
    """
    risk_contribs = _risk_contributions(weights, cov_matrix)
    n = len(weights)
    target_contrib = risk_contribs.mean()  # if equal, every RC should equal the average
    return np.sum((risk_contribs - target_contrib) ** 2)


def risk_parity_portfolio(cov_matrix: pd.DataFrame) -> pd.Series:
    """
    Solve for the Risk Parity Portfolio: weights such that every asset
    contributes equally to total portfolio risk.

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.

    Returns
    -------
    pd.Series
        Optimal weights indexed by ticker, summing to 1.
    """
    n = len(cov_matrix)
    tickers = cov_matrix.columns

    initial_guess = np.array([1 / n] * n)
    bounds = tuple((0.0001, 1) for _ in range(n))  # avoid exact zero (division stability)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    result = minimize(
        _risk_parity_objective,
        initial_guess,
        args=(cov_matrix.values,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return pd.Series(result.x, index=tickers)

def _negative_diversification_ratio(weights: np.ndarray, individual_vols: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Negative Diversification Ratio (negated for use with a minimizer).

    DR = (w . individual_vols) / portfolio_volatility
    """
    weighted_avg_vol = weights @ individual_vols
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    return -(weighted_avg_vol / port_vol)


def max_diversification_portfolio(cov_matrix: pd.DataFrame) -> pd.Series:
    """
    Solve for the Maximum Diversification Portfolio: weights that maximize
    the Diversification Ratio.

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Annualized covariance matrix.

    Returns
    -------
    pd.Series
        Optimal weights indexed by ticker, summing to 1.
    """
    n = len(cov_matrix)
    tickers = cov_matrix.columns
    individual_vols = np.sqrt(np.diag(cov_matrix))  # each asset's own volatility

    initial_guess = np.array([1 / n] * n)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    result = minimize(
        _negative_diversification_ratio,
        initial_guess,
        args=(individual_vols, cov_matrix.values),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return pd.Series(result.x, index=tickers)
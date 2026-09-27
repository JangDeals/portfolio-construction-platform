"""
Portfolio theory module.
Implements core Modern Portfolio Theory calculations from scratch:
expected return, variance, volatility, and Sharpe ratio for a portfolio
given asset weights.
"""

import numpy as np
import pandas as pd

# --- Portfolio Expected Return Calculation Function ---
def portfolio_expected_return(weights: np.ndarray, expected_returns: pd.Series) -> float:
    """
    Calculate expected portfolio return as the weighted average of asset
    expected returns.

    E(Rp) = sum(w_i * E(R_i))

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights, must sum to 1.
    expected_returns : pd.Series
        Expected (e.g. annualized) return for each asset.

    Returns
    -------
    float
        Expected portfolio return.
    """
    return np.dot(weights, expected_returns)

# --- Portfolio Variance Calculation Function ---
def portfolio_variance(weights: np.ndarray, covariance_matrix: pd.DataFrame) -> float:
    """
    Calculate portfolio variance using the matrix form:

    Var(Rp) = w^T . Sigma . w

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights, must sum to 1.
    covariance_matrix : pd.DataFrame
        Covariance matrix of asset returns (n x n).

    Returns
    -------
    float
        Portfolio variance.
    """
    return np.dot(weights.T, np.dot(covariance_matrix, weights))

def portfolio_volatility(weights: np.ndarray, covariance_matrix: pd.DataFrame) -> float:
    """
    Calculate portfolio volatility (standard deviation) from portfolio variance.

    Returns
    -------
    float
        Portfolio volatility.
    """
    return np.sqrt(portfolio_variance(weights, covariance_matrix))

# --- Portfolio Sharpe Ratio Calculation Function ---
def sharpe_ratio(portfolio_return: float, portfolio_vol: float, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio: excess return per unit of risk.

    Sharpe = (E(Rp) - Rf) / sigma_p

    Parameters
    ----------
    portfolio_return : float
        Expected (annualized) portfolio return.
    portfolio_vol : float
        Portfolio volatility (annualized).
    risk_free_rate : float, default 0.0
        Risk-free rate, same annualization basis as portfolio_return.

    Returns
    -------
    float
        Sharpe ratio.
    """
    return (portfolio_return - risk_free_rate) / portfolio_vol

# --- Portfolio Performance Summary Function ---
def calculate_annualized_covariance_matrix(simple_returns: pd.DataFrame, trading_days_per_year: int = 252) -> pd.DataFrame:
    """
    Calculate the annualized covariance matrix from daily simple returns.

    Parameters
    ----------
    simple_returns : pd.DataFrame
        Daily simple returns, dates as index, tickers as columns.
    trading_days_per_year : int, default 252

    Returns
    -------
    pd.DataFrame
        Annualized covariance matrix (n x n).
    """
    return simple_returns.cov() * trading_days_per_year



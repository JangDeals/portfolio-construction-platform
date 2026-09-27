"""
Portfolio recommendation engine.
Translates investor preferences and constraints into a concrete portfolio
recommendation, built on top of the optimization and construction engines
from Modules 11-14. Deliberately separates three concerns: investor
preferences, portfolio constraints, and quantitative calculations.
"""

import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from src.optimization.mean_variance import minimum_variance_portfolio, maximum_sharpe_portfolio
from src.portfolio.construction import equal_weight_portfolio
from src.risk.risk_engine import herfindahl_index


# --- Layer 1: Investor Preference Profiles ---
# Maps a human-readable risk profile to a strategy + constraint template.
# This is a POLICY layer, informed by findings from Modules 12-18, but
# contains no calculation logic itself.

RISK_PROFILES = {
    "Conservative": {
        "strategy": "Minimum Variance",
        "asset_class_bounds": {"Equity": (0.15, 0.35), "FixedIncome": (0.45, 0.70), "Alternatives": (0.05, 0.20)},
        "max_weight": 0.30,
        "rationale": (
            "Prioritizes capital preservation. Uses Minimum Variance, but "
            "constrained with a minimum equity floor and a fixed-income "
            "ceiling — per Module 18's finding, an UNCONSTRAINED Minimum "
            "Variance portfolio is dangerously exposed to inflation/rate "
            "shocks due to near-total bond concentration."
        ),
    },
    "Moderate": {
        "strategy": "Maximum Sharpe",
        "asset_class_bounds": {"Equity": (0.35, 0.60), "FixedIncome": (0.25, 0.45), "Alternatives": (0.10, 0.25)},
        "max_weight": 0.35,
        "rationale": (
            "Balances growth and stability. Uses Maximum Sharpe within "
            "asset-class bounds to avoid Module 12's corner-solution "
            "concentration, while still seeking meaningfully better "
            "risk-adjusted return than a naive equal-weight approach."
        ),
    },
    "Aggressive": {
        "strategy": "Maximum Sharpe",
        "asset_class_bounds": {"Equity": (0.55, 0.80), "FixedIncome": (0.10, 0.30), "Alternatives": (0.05, 0.20)},
        "max_weight": 0.45,
        "rationale": (
            "Prioritizes growth, accepting higher risk. Uses Maximum "
            "Sharpe with a higher equity floor. Per Module 16's finding, "
            "this profile should be RE-OPTIMIZED periodically (annually) "
            "rather than held as a frozen one-time estimate."
        ),
    },
}


# The engine that connects preferences to calculations
def recommend_portfolio(
    risk_profile: str,
    annualized_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    tickers: list[str],
    asset_class_map: dict,
    risk_free_rate: float = 0.02,
    custom_overrides: dict = None,
) -> dict:
    """
    Generate a portfolio recommendation for a given investor risk profile.

    Parameters
    ----------
    risk_profile : str
        One of "Conservative", "Moderate", "Aggressive".
    annualized_returns : pd.Series
    cov_matrix : pd.DataFrame
    tickers : list[str]
    asset_class_map : dict
    risk_free_rate : float, default 0.02
    custom_overrides : dict, optional
        Allows overriding any of the profile's default settings, e.g.
        {"max_weight": 0.25} to impose a tighter concentration cap than
        the profile's default. This is the "flexible" layer (B) sitting
        underneath the simple labeled profiles (A).

    Returns
    -------
    dict
        weights, hhi, recommended_rebalance_frequency, rationale.
    """
    if risk_profile not in RISK_PROFILES:
        raise ValueError(f"Unknown risk profile: {risk_profile}. Choose from {list(RISK_PROFILES.keys())}")

    profile = dict(RISK_PROFILES[risk_profile])  # copy, so overrides don't mutate the template
    if custom_overrides:
        profile.update(custom_overrides)

    # --- Layer 3: Quantitative calculation (calls existing, tested engines) ---
    if profile["strategy"] == "Minimum Variance":
        weights = minimum_variance_portfolio(
            cov_matrix,
            max_weight=profile["max_weight"],
            asset_class_map=asset_class_map,
            asset_class_bounds=profile["asset_class_bounds"],
            max_hhi=profile.get("max_hhi"),
        )
    elif profile["strategy"] == "Maximum Sharpe":
        weights = maximum_sharpe_portfolio(
            annualized_returns, cov_matrix, risk_free_rate,
            max_weight=profile["max_weight"],
            asset_class_map=asset_class_map,
            asset_class_bounds=profile["asset_class_bounds"],
            max_hhi=profile.get("max_hhi"),
        )
    else:
        raise ValueError(f"Unsupported strategy in profile: {profile['strategy']}")

    # --- Rebalancing frequency recommendation, using Module 17's finding ---
    hhi = herfindahl_index(weights.values)
    if hhi > 0.30:
        rebalance_frequency = "Quarterly (high concentration — monitor closely, per Module 17 finding)"
    elif hhi > 0.15:
        rebalance_frequency = "Semi-Annually"
    else:
        rebalance_frequency = "Annually"

    return {
        "risk_profile": risk_profile,
        "weights": weights,
        "hhi": hhi,
        "recommended_rebalance_frequency": rebalance_frequency,
        "rationale": profile["rationale"],
    }
    
    

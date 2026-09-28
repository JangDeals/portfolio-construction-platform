# streamlit_app/pages_content/construction.py
"""
Portfolio Construction page: runs the recommendation engine using the
user's selected risk profile, and displays the resulting portfolio.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))  # project root

from src.utils.paths import RAW_DATA_DIR
from src.features.return_calculator import calculate_simple_returns
from src.portfolio.portfolio_theory import calculate_annualized_covariance_matrix
from src.portfolio.recommendation_engine import recommend_portfolio


@st.cache_data
def load_data_and_compute_inputs():
    """
    Load saved price data and compute the annualized returns and covariance
    matrix needed for optimization. Cached so this only runs once per app
    session, not on every rerun.
    """
    prices = pd.read_csv(RAW_DATA_DIR / "asset_prices.csv", index_col=0, parse_dates=True)
    simple_returns = calculate_simple_returns(prices)
    annualized_returns = (1 + simple_returns).prod() ** (252 / len(simple_returns)) - 1
    cov_matrix = calculate_annualized_covariance_matrix(simple_returns)
    return prices, simple_returns, annualized_returns, cov_matrix


ASSET_CLASS_MAP = {
    "SPY": "Equity", "QQQ": "Equity", "VEA": "Equity", "VWO": "Equity",
    "IEF": "FixedIncome", "TLT": "FixedIncome", "SHY": "FixedIncome",
    "VNQ": "Alternatives", "GLD": "Alternatives",
}


def render_construction_page():
    st.title("Portfolio Construction")

    if st.session_state["risk_profile"] is None:
        st.error("Please select a risk profile first.")
        if st.button("← Back to Setup"):
            st.session_state["current_step"] = "portfolio_setup"
            st.rerun()
        return

    with st.spinner("Loading data and computing portfolio..."):
        prices, simple_returns, annualized_returns, cov_matrix = load_data_and_compute_inputs()
        tickers = st.session_state["selected_tickers"]

        result = recommend_portfolio(
            st.session_state["risk_profile"],
            annualized_returns, cov_matrix, tickers, ASSET_CLASS_MAP,
        )

    st.session_state["portfolio_weights"] = result["weights"]

    st.session_state["recommendation"] = result
    # Invalidate results from later pages, so a profile change can never
    # leave stale numbers behind for the report to pick up.
    st.session_state["oos_summary"] = None
    st.session_state["in_sample_metrics"] = None
    
    st.subheader(f"Recommended Portfolio: {result['risk_profile']}")
    st.caption(result["rationale"])

    weights_df = result["weights"].reset_index()
    weights_df.columns = ["Ticker", "Weight"]
    weights_df = weights_df[weights_df["Weight"] > 0.001]  # hide near-zero noise from optimizer

    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(weights_df.set_index("Ticker"))
    with col2:
        st.dataframe(weights_df.style.format({"Weight": "{:.1%}"}), hide_index=True)

    st.metric("Concentration (HHI)", f"{result['hhi']:.3f}")
    st.metric("Recommended Rebalancing", result["recommended_rebalance_frequency"])

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state["current_step"] = "portfolio_setup"
            st.rerun()
    with col2:
        if st.button("Continue to Analysis →", type="primary"):
            st.session_state["current_step"] = "analysis"
            st.rerun()
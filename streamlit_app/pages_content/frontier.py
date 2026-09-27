# streamlit_app/pages_content/frontier.py
"""
Efficient Frontier page: visualizes the frontier with the current
portfolio's own risk/return point marked on it.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.optimization.mean_variance import build_efficient_frontier, _portfolio_variance
from src.portfolio.portfolio_theory import portfolio_expected_return, portfolio_volatility
from pages_content.construction import load_data_and_compute_inputs


@st.cache_data
def get_efficient_frontier(annualized_returns, cov_matrix):
    """
    Cached wrapper around build_efficient_frontier, since this involves
    50 separate optimization solves and should not be recomputed on every
    page interaction.
    """
    return build_efficient_frontier(annualized_returns, cov_matrix, n_points=50)


def render_frontier_page():
    st.title("Efficient Frontier")

    if st.session_state["portfolio_weights"] is None:
        st.error("Please construct a portfolio first.")
        if st.button("← Back to Construction"):
            st.session_state["current_step"] = "construction"
            st.rerun()
        return

    weights = st.session_state["portfolio_weights"]
    prices, simple_returns, annualized_returns, cov_matrix = load_data_and_compute_inputs()

    with st.spinner("Computing efficient frontier (50 optimizations)..."):
        frontier = get_efficient_frontier(annualized_returns, cov_matrix)

    port_return = portfolio_expected_return(weights.values, annualized_returns)
    port_vol = portfolio_volatility(weights.values, cov_matrix)

    chart_data = pd.DataFrame({
        "Volatility": frontier["volatility"],
        "Frontier Return": frontier["target_return"],
    })

    st.line_chart(chart_data.set_index("Volatility"))

    st.markdown(
        f"**Your Portfolio** sits at **{port_vol:.2%} volatility** and "
        f"**{port_return:.2%} expected return**."
    )

    # Show how close the portfolio is to the frontier at a similar volatility level
    closest_idx = (frontier["volatility"] - port_vol).abs().idxmin()
    frontier_return_at_similar_vol = frontier.loc[closest_idx, "target_return"]
    gap = frontier_return_at_similar_vol - port_return

    if gap > 0.001:
        st.info(
            f"At a similar volatility level (~{frontier['volatility'][closest_idx]:.2%}), "
            f"the efficient frontier achieves {frontier_return_at_similar_vol:.2%} return — "
            f"a gap of {gap:.2%}. This is expected: your portfolio is built under "
            f"asset-class and concentration constraints (Module 13), which trade "
            f"some theoretical efficiency for a more realistic, diversified allocation."
        )
    else:
        st.success("Your portfolio sits at or very near the efficient frontier at this risk level.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state["current_step"] = "analysis"
            st.rerun()
    with col2:
        if st.button("Continue →", type="primary"):
            st.session_state["current_step"] = "rebalancing"
            st.rerun()
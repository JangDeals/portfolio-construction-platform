# streamlit_app/pages_content/analysis.py
"""
Analysis page: displays risk and performance metrics for the constructed
portfolio, and (in Part 2) the efficient frontier with the portfolio marked.
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.portfolio.portfolio_theory import portfolio_expected_return, portfolio_volatility, sharpe_ratio
from src.risk.risk_engine import (
    downside_volatility, maximum_drawdown, historical_var, historical_cvar, herfindahl_index
)
from src.performance.performance_analytics import calculate_cagr, sortino_ratio, calmar_ratio
from src.features.return_calculator import calculate_cumulative_returns
from pages_content.construction import load_data_and_compute_inputs


def render_analysis_page():
    st.title("Risk & Performance Analysis")

    if st.session_state["portfolio_weights"] is None:
        st.error("Please construct a portfolio first.")
        if st.button("← Back to Construction"):
            st.session_state["current_step"] = "construction"
            st.rerun()
        return

    weights = st.session_state["portfolio_weights"]
    prices, simple_returns, annualized_returns, cov_matrix = load_data_and_compute_inputs()

    portfolio_returns = simple_returns.dot(weights.values)
    portfolio_returns.name = "Portfolio"
    portfolio_cumulative = calculate_cumulative_returns(portfolio_returns)

    exp_return = portfolio_expected_return(weights.values, annualized_returns)
    exp_vol = portfolio_volatility(weights.values, cov_matrix)
    sharpe = sharpe_ratio(exp_return, exp_vol, risk_free_rate=0.02)
    d_vol = downside_volatility(portfolio_returns)
    mdd = maximum_drawdown(portfolio_cumulative)
    var_95 = historical_var(portfolio_returns, 0.95)
    cvar_95 = historical_cvar(portfolio_returns, 0.95)
    cagr = calculate_cagr(portfolio_cumulative)
    sortino = sortino_ratio(exp_return, d_vol, risk_free_rate=0.02)
    calmar = calmar_ratio(exp_return, mdd["max_drawdown"])

    st.subheader("Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Return", f"{exp_return:.2%}")
    col2.metric("Volatility", f"{exp_vol:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.3f}")
    col4.metric("CAGR", f"{cagr:.2%}")

    st.subheader("Risk Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Drawdown", f"{mdd['max_drawdown']:.2%}")
    col2.metric("Historical VaR (95%)", f"{var_95:.2%}")
    col3.metric("Historical CVaR (95%)", f"{cvar_95:.2%}")
    col4.metric("Downside Volatility", f"{d_vol:.2%}")

    st.subheader("Risk-Adjusted Ratios")
    col1, col2 = st.columns(2)
    col1.metric("Sortino Ratio", f"{sortino:.3f}")
    col2.metric("Calmar Ratio", f"{calmar:.3f}")

    st.subheader("Cumulative Performance")
    st.line_chart(portfolio_cumulative)

    st.caption(
        f"Maximum drawdown occurred between {mdd['peak_date']} and "
        f"{mdd['trough_date']}."
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state["current_step"] = "construction"
            st.rerun()
    with col2:
        if st.button("Continue to Frontier →", type="primary"):
            st.session_state["current_step"] = "frontier"
            st.rerun()
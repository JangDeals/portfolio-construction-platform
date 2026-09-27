# streamlit_app/pages_content/rebalancing.py
"""
Rebalancing page: simulates portfolio drift over a historical illustration
window and shows the rebalancing trades required to return to target.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.backtesting.rebalancing_engine import compute_drifted_weights, compute_rebalancing_trades
from src.risk.risk_engine import herfindahl_index
from pages_content.construction import load_data_and_compute_inputs


def render_rebalancing_page():
    st.title("Rebalancing Analysis")

    if st.session_state["portfolio_weights"] is None:
        st.error("Please construct a portfolio first.")
        if st.button("← Back to Construction"):
            st.session_state["current_step"] = "construction"
            st.rerun()
        return

    weights = st.session_state["portfolio_weights"]
    prices, simple_returns, annualized_returns, cov_matrix = load_data_and_compute_inputs()

    st.info(
        "This is an illustrative simulation: it shows how YOUR recommended "
        "weights would have drifted had they been held, untouched, since "
        "3 years ago. It is not this portfolio's actual trading history, "
        "since it was just constructed."
    )

    illustration_start = (prices.index.max() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    drifted = compute_drifted_weights(weights, prices, start_date=illustration_start)
    current_actual_weights = drifted.iloc[-1]

    comparison_df = pd.DataFrame({
        "Target Weight": weights,
        "Current (Drifted) Weight": current_actual_weights,
    })
    comparison_df["Drift"] = comparison_df["Current (Drifted) Weight"] - comparison_df["Target Weight"]
    comparison_df = comparison_df[comparison_df["Target Weight"] > 0.001]

    st.subheader(f"Drift Since {illustration_start}")
    st.dataframe(
        comparison_df.style.format("{:.2%}").background_gradient(
            subset=["Drift"], cmap="RdYlGn_r", vmin=-0.10, vmax=0.10
        )
    )

    hhi_current = herfindahl_index(current_actual_weights.values)
    st.metric("Current Concentration (HHI)", f"{hhi_current:.3f}", 
              delta=f"{hhi_current - herfindahl_index(weights.values):.3f} vs target")

    st.subheader("Recommended Rebalancing Trades")
    trade_result = compute_rebalancing_trades(current_actual_weights, weights)
    trades_df = trade_result["trades"][trade_result["trades"].abs() > 0.001].to_frame("Trade (Weight Change)")
    trades_df["Action"] = trades_df["Trade (Weight Change)"].apply(lambda x: "BUY" if x > 0 else "SELL")

    st.dataframe(trades_df.style.format({"Trade (Weight Change)": "{:+.2%}"}))
    st.metric("Total Turnover Required", f"{trade_result['turnover']:.2%}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state["current_step"] = "frontier"
            st.rerun()
    with col2:
        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
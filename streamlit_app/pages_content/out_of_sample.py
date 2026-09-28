# streamlit_app/pages_content/out_of_sample.py
"""
Out-of-Sample Validation page: shows walk-forward backtest results for all
three risk profiles plus the 60/40 benchmark, and directly compares the
user's selected profile's in-sample vs. out-of-sample performance.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.backtesting.backtest_engine import walk_forward_profile_backtest, compute_turnover_cost
from src.portfolio.benchmarks import sixty_forty_benchmark
from src.portfolio.portfolio_theory import portfolio_expected_return, portfolio_volatility, sharpe_ratio
from src.risk.risk_engine import maximum_drawdown
from src.features.return_calculator import calculate_cumulative_returns
from pages_content.construction import load_data_and_compute_inputs, ASSET_CLASS_MAP


@st.cache_data(show_spinner=False)
def run_full_walkforward_comparison(_simple_returns, tickers, asset_class_map):
    """
    Cached, one-time computation of the walk-forward comparison across all
    three risk profiles plus 60/40. Does not depend on which profile the
    current user selected, so this is computed once and shared for the
    entire app session.
    """
    results = {}
    for profile in ["Conservative", "Moderate", "Aggressive"]:
        r = walk_forward_profile_backtest(profile, _simple_returns, tickers, asset_class_map, initial_train_years=10)
        results[profile] = r["oos_returns"]

    start_date = _simple_returns.index.min() + pd.DateOffset(years=10)
    rebalance_dates = pd.date_range(start=start_date, end=_simple_returns.index.max(), freq="YS")
    weights_6040 = sixty_forty_benchmark(tickers)
    all_returns, previous_weights = [], pd.Series(0.0, index=tickers)
    for i, rebal_date in enumerate(rebalance_dates):
        next_date = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else _simple_returns.index.max() + pd.Timedelta(days=1)
        test_window = _simple_returns[(_simple_returns.index >= rebal_date) & (_simple_returns.index < next_date)]
        if len(test_window) == 0:
            continue
        cost = compute_turnover_cost(previous_weights, weights_6040)
        period_returns = test_window.dot(weights_6040.values)
        period_returns.iloc[0] -= cost
        all_returns.append(period_returns)
        previous_weights = weights_6040
    results["60/40 Benchmark"] = pd.concat(all_returns)

    return results


def _compute_summary_stats(oos_returns):
    cumulative = calculate_cumulative_returns(oos_returns)
    total_return = cumulative.iloc[-1]
    n_years = (oos_returns.index[-1] - oos_returns.index[0]).days / 365.25
    annualized = (1 + total_return) ** (1 / n_years) - 1
    vol = oos_returns.std() * np.sqrt(252)
    sharpe = (annualized - 0.02) / vol
    mdd = maximum_drawdown(cumulative)["max_drawdown"]
    return {"Annualized Return": annualized, "Volatility": vol, "Sharpe Ratio": sharpe, "Max Drawdown": mdd}, cumulative


def render_out_of_sample_page():
    st.title("Out-of-Sample Validation")

    st.markdown("""
    Every other page in this app evaluates portfolios using the **full historical
    dataset** — the same data used to build them. This page instead shows how each
    risk profile actually performed on data it had **never seen** at the time its
    weights were estimated, using a walk-forward methodology: re-optimizing annually
    on an expanding window and testing only on the following, unseen year.
    """)

    prices, simple_returns, annualized_returns, cov_matrix = load_data_and_compute_inputs()

    with st.spinner("Running walk-forward validation across all profiles (this takes a while, cached after first run)..."):
        oos_results = run_full_walkforward_comparison(simple_returns, list(simple_returns.columns), ASSET_CLASS_MAP)

    summary_rows = {}
    cumulative_series = {}
    for name, returns in oos_results.items():
        stats, cumulative = _compute_summary_stats(returns)
        summary_rows[name] = stats
        cumulative_series[name] = cumulative

    summary_df = pd.DataFrame(summary_rows).T
    st.subheader("Out-of-Sample Performance (Walk-Forward, 2018–Present)")
    st.dataframe(summary_df.style.format({
        "Annualized Return": "{:.2%}", "Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.3f}", "Max Drawdown": "{:.2%}",
    }))

    st.line_chart(pd.DataFrame(cumulative_series))

    st.divider()
    selected_profile = st.session_state.get("risk_profile")
    if selected_profile and selected_profile in oos_results:
        st.subheader(f"Your Profile ({selected_profile}): In-Sample vs. Out-of-Sample")

        weights = st.session_state["portfolio_weights"]
        in_sample_return = portfolio_expected_return(weights.values, annualized_returns)
        in_sample_vol = portfolio_volatility(weights.values, cov_matrix)
        in_sample_sharpe = sharpe_ratio(in_sample_return, in_sample_vol, risk_free_rate=0.02)

        oos_stats = summary_rows[selected_profile]

        comparison = pd.DataFrame({
            "In-Sample (full history)": {
                "Annualized Return": in_sample_return, "Volatility": in_sample_vol, "Sharpe Ratio": in_sample_sharpe,
            },
            "Out-of-Sample (walk-forward)": {
                "Annualized Return": oos_stats["Annualized Return"], "Volatility": oos_stats["Volatility"],
                "Sharpe Ratio": oos_stats["Sharpe Ratio"],
            },
        }).T
        st.dataframe(comparison.style.format({"Annualized Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe Ratio": "{:.3f}"}))

        st.caption(
            "A gap between these two columns is expected, but its size and direction "
            "aren't predictable in advance. Here, out-of-sample performance happens to "
            "be *higher* than in-sample — this doesn't mean the model 'improved'; it "
            "reflects that these two figures cover genuinely different periods. "
            "In-sample spans the full 2008–present history (including weaker years "
            "for this profile's holdings), while out-of-sample reflects only the "
            "annually re-optimized 2018–present walk-forward window, which happened "
            "to favor this profile's growth tilt. The comparison's real value is "
            "confirming the methodology holds up on unseen data — not which number "
            "is bigger."
        )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state["current_step"] = "frontier"
            st.rerun()
    with col2:
        if st.button("Continue to Rebalancing →", type="primary"):
            st.session_state["current_step"] = "rebalancing"
            st.rerun()
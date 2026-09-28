"""
Session state initialization and helper functions for the Streamlit app.
Centralizes all session_state keys so their existence and defaults are
defined in exactly one place.
"""

import streamlit as st


def initialize_session_state():
    """
    Initialize every session_state key used across the app, with sensible
    defaults. Must be called once at the top of app.py, before any page
    content is rendered.
    """
    defaults = {
        "current_step": "landing",
        "selected_tickers": None,
        "risk_profile": None,
        "custom_constraints": None,
        "selected_method": None,
        "portfolio_weights": None,
        "backtest_results": None,
        "recommendation": None,
        "oos_summary": None,
        "in_sample_metrics": None,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def go_to_step(step_name: str):
    """Navigate to a given step and trigger a rerun."""
    st.session_state["current_step"] = step_name
    st.rerun()
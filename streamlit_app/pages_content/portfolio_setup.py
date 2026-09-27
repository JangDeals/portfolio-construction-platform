"""
Portfolio Setup page: risk profile selection and universe confirmation.
"""

import streamlit as st


def render_portfolio_setup_page():
    st.title("Portfolio Setup")

    st.subheader("1. Select Your Risk Profile")
    risk_profile = st.radio(
        "Choose the profile that best matches your investment objective:",
        options=["Conservative", "Moderate", "Aggressive"],
        index=None,
        help=(
            "Conservative: prioritizes capital preservation. "
            "Moderate: balances growth and stability. "
            "Aggressive: prioritizes growth, accepting higher risk."
        ),
    )

    if risk_profile:
        st.session_state["risk_profile"] = risk_profile
        st.success(f"Selected: {risk_profile}")

    st.subheader("2. Asset Universe")
    default_tickers = ["SPY", "QQQ", "VEA", "VWO", "IEF", "TLT", "SHY", "VNQ", "GLD"]
    st.write("This platform uses the following 9-asset universe:")
    st.code(", ".join(default_tickers))
    st.session_state["selected_tickers"] = default_tickers

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state["current_step"] = "landing"
            st.rerun()
    with col2:
        if st.session_state["risk_profile"] is not None:
            if st.button("Continue →", type="primary"):
                st.session_state["current_step"] = "construction"
                st.rerun()
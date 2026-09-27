"""
Landing page: introduces the platform and starts the wizard flow.
"""

import streamlit as st


def render_landing_page():
    st.title("Multi-Asset Portfolio Construction Platform")
    st.markdown("""
    Build, optimize, and analyze a diversified multi-asset portfolio using
    Modern Portfolio Theory, quantitative risk management, and historically
    validated construction methods.

    This platform will walk you through:
    - Selecting your asset universe and risk profile
    - Constructing and optimizing a portfolio suited to your objectives
    - Analyzing its risk, performance, and diversification
    - Understanding how it would have behaved historically, including
      real market crises
    - Generating rebalancing recommendations to keep it aligned over time
    """)

    st.info(
        "This tool is for educational and research purposes. It does not "
        "constitute financial advice."
    )

    if st.button("Get Started", type="primary"):
        st.session_state["current_step"] = "portfolio_setup"
        st.rerun()
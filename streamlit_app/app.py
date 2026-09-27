"""
Main Streamlit application entry point.
Orchestrates the step-based wizard flow, routing to the correct page
content based on session_state["current_step"].
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # project root
sys.path.append(str(Path(__file__).resolve().parent))       # streamlit_app/ itself


from utils.state_helpers import initialize_session_state
from pages_content.landing import render_landing_page
from pages_content.portfolio_setup import render_portfolio_setup_page
from pages_content.construction import render_construction_page
from pages_content.analysis import render_analysis_page
from pages_content.frontier import render_frontier_page
from pages_content.rebalancing import render_rebalancing_page

st.set_page_config(page_title="Portfolio Construction Platform", layout="wide")

initialize_session_state()

step = st.session_state["current_step"]

if step == "landing":
    render_landing_page()   
elif step == "portfolio_setup":
    render_portfolio_setup_page()
elif step == "construction":
    render_construction_page()
elif step == "analysis":
    render_analysis_page()
elif step == "frontier":
    render_frontier_page()
elif step == "rebalancing":
    render_rebalancing_page()
elif step == "rebalancing":
    st.write("Rebalancing — coming in the next step of this module")
else:
    st.error(f"Unknown step: {step}")
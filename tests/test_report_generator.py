"""
Tests for src/utils/report_generator.py.
Relies on the editable install (pip install -e .) rather than a sys.path hack.
"""

import pandas as pd
from src.utils.report_generator import build_portfolio_report


def _base_kwargs():
    return dict(
        risk_profile="Moderate",
        rationale="Test rationale.",
        weights=pd.Series({"QQQ": 0.35, "IEF": 0.35, "SHY": 0.09, "GLD": 0.21, "SPY": 0.0}),
        hhi=0.297,
        rebalance_frequency="Semi-Annually",
        data_start="2008-01-02",
        data_end="2026-09-25",
    )


def _stats(ret, vol, sharpe, dd):
    return {"Annualized Return": ret, "Volatility": vol, "Sharpe Ratio": sharpe, "Max Drawdown": dd}


def test_report_contains_core_content_and_disclaimer():
    report = build_portfolio_report(**_base_kwargs())
    assert "# Portfolio Recommendation Report" in report
    assert "not financial advice" in report
    assert "Moderate" in report
    assert "QQQ" in report


def test_report_omits_zero_weight_positions():
    report = build_portfolio_report(**_base_kwargs())
    assert "| SPY |" not in report


def test_report_handles_missing_optional_sections():
    """No in-sample, out-of-sample, or drift data: must not crash."""
    report = build_portfolio_report(**_base_kwargs())
    assert "Not available for this session" in report


def test_report_marks_selected_profile_in_out_of_sample_table():
    oos = {
        "Conservative": _stats(0.06, 0.06, 0.66, -0.16),
        "Moderate": _stats(0.12, 0.11, 0.88, -0.27),
        "60/40 Benchmark": _stats(0.11, 0.12, 0.77, -0.21),
    }
    report = build_portfolio_report(**_base_kwargs(), out_of_sample=oos)
    assert "**Moderate (your profile)**" in report
    assert "**Conservative (your profile)**" not in report
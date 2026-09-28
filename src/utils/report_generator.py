"""
Portfolio report generator.
Formats results already computed elsewhere in the app into a single
Markdown report. Contains no calculation logic of its own.
"""

from datetime import datetime
import pandas as pd


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    """Build a Markdown table by hand (avoids the optional 'tabulate' dependency)."""
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_portfolio_report(
    risk_profile: str,
    rationale: str,
    weights: pd.Series,
    hhi: float,
    rebalance_frequency: str,
    data_start: str,
    data_end: str,
    in_sample: dict | None = None,
    out_of_sample: dict | None = None,
    drift_table: pd.DataFrame | None = None,
    drift_start: str | None = None,
    trades_table: pd.DataFrame | None = None,
    turnover: float | None = None,
) -> str:
    """
    Build a Markdown portfolio report.

    Parameters
    ----------
    risk_profile, rationale, hhi, rebalance_frequency
        From the recommendation engine's output.
    weights : pd.Series
        Target weights indexed by ticker.
    data_start, data_end : str
        Date range of the historical data used.
    in_sample : dict, optional
        {"Annualized Return", "Volatility", "Sharpe Ratio"} for this portfolio.
    out_of_sample : dict, optional
        {profile_name: {"Annualized Return", "Volatility", "Sharpe Ratio",
        "Max Drawdown"}} for every profile plus the 60/40 benchmark.
    drift_table, trades_table : pd.DataFrame, optional
        The illustrative drift and rebalancing-trade tables from the app.
    drift_start : str, optional
        Start date of the drift illustration.
    turnover : float, optional

    Returns
    -------
    str
        The complete report as Markdown text.
    """
    lines = []

    lines += [
        "# Portfolio Recommendation Report",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*  ",
        f"*Historical data used: {data_start} to {data_end}*",
        "",
        "> **For educational and research purposes only. This report is not financial advice.**",
        "",
    ]

    # --- 1. Recommended portfolio ---
    held = weights[weights > 0.001].sort_values(ascending=False)
    lines += [
        f"## 1. Recommended Portfolio: {risk_profile}",
        "",
        rationale,
        "",
        _md_table(["Ticker", "Weight"], [[t, f"{w:.1%}"] for t, w in held.items()]),
        "",
        "*Positions below 0.1% are omitted.*",
        "",
        f"- **Concentration (HHI):** {hhi:.3f}",
        f"- **Suggested rebalancing frequency:** {rebalance_frequency}",
        "",
    ]

    # --- 2. In-sample metrics ---
    lines += ["## 2. In-Sample Metrics (illustrative only)", ""]
    if in_sample:
        lines += [
            "These figures apply the recommended weights to the full historical "
            "dataset, which is the same data used to estimate them. They are an "
            "illustration, not a forecast.",
            "",
            _md_table(
                ["Annualized Return", "Volatility", "Sharpe Ratio"],
                [[f"{in_sample['Annualized Return']:.2%}",
                  f"{in_sample['Volatility']:.2%}",
                  f"{in_sample['Sharpe Ratio']:.3f}"]],
            ),
            "",
        ]
    else:
        lines += ["_Not available for this session._", ""]

    # --- 3. Out-of-sample validation ---
    lines += ["## 3. Out-of-Sample Validation (walk-forward)", ""]
    if out_of_sample:
        rows = []
        for name, s in out_of_sample.items():
            label = f"**{name} (your profile)**" if name == risk_profile else name
            rows.append([
                label,
                f"{s['Annualized Return']:.2%}",
                f"{s['Volatility']:.2%}",
                f"{s['Sharpe Ratio']:.3f}",
                f"{s['Max Drawdown']:.2%}",
            ])
        lines += [
            "Weights were re-estimated each January using only prior data (10-year "
            "minimum training window), then tested on the following, unseen year. "
            "Transaction costs are modeled at 10 bps per unit of turnover.",
            "",
            _md_table(
                ["Strategy", "Annualized Return", "Volatility", "Sharpe Ratio", "Max Drawdown"],
                rows,
            ),
            "",
        ]
    else:
        lines += ["_Not available for this session._", ""]

    # --- 4. Rebalancing illustration ---
    lines += ["## 4. Rebalancing Illustration (hypothetical)", ""]
    if drift_table is not None and trades_table is not None:
        lines += [
            f"This shows how the recommended weights would have drifted had they been "
            f"held untouched since {drift_start}. It is not this portfolio's actual "
            f"trading history.",
            "",
            _md_table(
                ["Ticker", "Target", "Current (Drifted)", "Drift"],
                [[t, f"{r['Target Weight']:.2%}", f"{r['Current (Drifted) Weight']:.2%}",
                  f"{r['Drift']:+.2%}"] for t, r in drift_table.iterrows()],
            ),
            "",
            "**Trades required to return to target:**",
            "",
            _md_table(
                ["Ticker", "Action", "Weight Change"],
                [[t, r["Action"], f"{r['Trade (Weight Change)']:+.2%}"]
                 for t, r in trades_table.iterrows()],
            ),
            "",
        ]
        if turnover is not None:
            lines += [f"**Total turnover required:** {turnover:.2%}", ""]
    else:
        lines += ["_Not available for this session._", ""]

    # --- Limitations ---
    lines += [
        "## Limitations",
        "",
        "- In-sample figures are evaluated on the same history used to build the portfolio.",
        "- The out-of-sample results cover one realized historical path (2019 onward). "
        "The risk-profile constraint templates were designed with knowledge of the full "
        "historical record, so even these results are not a pristine holdout.",
        "- Costs are modeled simply (10 bps per unit of turnover). Taxes, bid-ask spreads, "
        "and market impact are not modeled.",
        "- Expected returns are estimated from historical averages, which are noisy "
        "and may not persist.",
        "- Past performance does not guarantee future results.",
        "",
    ]

    return "\n".join(lines)
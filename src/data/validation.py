"""
Data validation module.
Runs quality checks on downloaded market data and produces a validation report.
"""

import pandas as pd
import numpy as np


def check_dimensions(prices: pd.DataFrame, expected_tickers: list[str]) -> dict:
    """Verify the DataFrame has the expected tickers as columns."""
    actual_tickers = set(prices.columns)
    expected_set = set(expected_tickers)

    missing = expected_set - actual_tickers
    extra = actual_tickers - expected_set

    return {
        "check": "dimensions",
        "passed": len(missing) == 0 and len(extra) == 0,
        "num_rows": len(prices),
        "num_columns": len(prices.columns),
        "missing_tickers": list(missing),
        "extra_tickers": list(extra),
    }

# Duplicate dates check
def check_duplicate_dates(prices: pd.DataFrame) -> dict:
    """Verify there are no duplicate index (date) values."""
    duplicate_dates = prices.index[prices.index.duplicated()].tolist()

    return {
        "check": "duplicate_dates",
        "passed": len(duplicate_dates) == 0,
        "num_duplicates": len(duplicate_dates),
        "duplicate_dates": [str(d.date()) for d in duplicate_dates],
    }
    
# Check for missing values
def check_missing_values(prices: pd.DataFrame) -> dict:
    """Check for NaN values in each ticker's column."""
    nan_counts = prices.isna().sum()
    tickers_with_nans = nan_counts[nan_counts > 0].to_dict()

    return {
        "check": "missing_values",
        "passed": len(tickers_with_nans) == 0,
        "nan_counts_by_ticker": tickers_with_nans,
        "total_nans": int(nan_counts.sum()),
    }

# Check for unusually large gaps between consecutive trading dates
def check_date_continuity(prices: pd.DataFrame, max_gap_days: int = 5) -> dict:
    """
    Check for unusually large gaps between consecutive trading dates.
    A gap larger than max_gap_days (default 5, covering long weekends/holidays)
    may indicate a data problem rather than a normal non-trading period.
    """
    date_diffs = prices.index.to_series().diff().dt.days
    large_gaps = date_diffs[date_diffs > max_gap_days]

    gap_details = [
        {"date": str(idx.date()), "gap_days": int(gap)}
        for idx, gap in large_gaps.items()
    ]

    return {
        "check": "date_continuity",
        "passed": len(gap_details) == 0,
        "num_large_gaps": len(gap_details),
        "gap_details": gap_details,
    }
    
# Check for suspicious price values
def check_suspicious_values(prices: pd.DataFrame, max_daily_change: float = 0.5) -> dict:
    """
    Flag zero/negative prices (impossible for an ETF) and single-day price
    changes larger than max_daily_change (default 50%), which likely indicate
    a data error rather than a genuine market move.
    """
    issues = {}

    # Zero or negative prices
    invalid_prices = (prices <= 0).sum()
    tickers_with_invalid = invalid_prices[invalid_prices > 0].to_dict()

    # Extreme single-day returns
    daily_returns = prices.pct_change()
    extreme_moves = {}
    for ticker in prices.columns:
        big_jumps = daily_returns[ticker][daily_returns[ticker].abs() > max_daily_change]
        if len(big_jumps) > 0:
            extreme_moves[ticker] = [
                {"date": str(idx.date()), "pct_change": round(float(val), 4)}
                for idx, val in big_jumps.items()
            ]

    return {
        "check": "suspicious_values",
        "passed": len(tickers_with_invalid) == 0 and len(extreme_moves) == 0,
        "tickers_with_zero_or_negative_prices": tickers_with_invalid,
        "extreme_daily_moves": extreme_moves,
    }

# Check for unusually high number of zero-volume days
def check_volume_anomalies(volumes: pd.DataFrame, zero_volume_threshold: int = 5) -> dict:
    """
    Flag tickers with an unusually high number of zero-volume days, which can
    indicate trading halts, delisting periods, or a data feed problem.
    """
    zero_volume_counts = (volumes == 0).sum()
    tickers_with_zero_volume = zero_volume_counts[zero_volume_counts > zero_volume_threshold].to_dict()

    return {
        "check": "volume_anomalies",
        "passed": len(tickers_with_zero_volume) == 0,
        "zero_volume_day_counts": tickers_with_zero_volume,
    }

# Run all checks and produce a combined report
def run_all_checks(prices: pd.DataFrame, volumes: pd.DataFrame, expected_tickers: list[str]) -> dict:
    """
    Run all data quality checks and return a combined report.
    """
    results = {
        "dimensions": check_dimensions(prices, expected_tickers),
        "duplicate_dates": check_duplicate_dates(prices),
        "missing_values": check_missing_values(prices),
        "date_continuity": check_date_continuity(prices),
        "suspicious_values": check_suspicious_values(prices),
        "volume_anomalies": check_volume_anomalies(volumes),
    }

    all_passed = all(result["passed"] for result in results.values())

    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    for name, result in results.items():
        status = "PASS" if result["passed"] else "FLAGGED"
        print(f"[{status}] {name}")
    print("=" * 60)
    print(f"Overall: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FLAGGED — review details'}")
    print("=" * 60)

    return {"all_passed": all_passed, "results": results}
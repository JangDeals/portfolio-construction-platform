"""
Centralized configuration.
Single source of truth for values used across multiple modules, so a
change here propagates everywhere rather than requiring edits in many
scattered files.
"""

from datetime import datetime

# --- Asset universe ---
TICKERS = ["SPY", "QQQ", "VEA", "VWO", "IEF", "TLT", "SHY", "VNQ", "GLD"]

ASSET_CLASS_MAP = {
    "SPY": "Equity", "QQQ": "Equity", "VEA": "Equity", "VWO": "Equity",
    "IEF": "FixedIncome", "TLT": "FixedIncome", "SHY": "FixedIncome",
    "VNQ": "Alternatives", "GLD": "Alternatives",
}

# --- Data acquisition ---
START_DATE = "2008-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# --- Financial assumptions ---
RISK_FREE_RATE = 0.02
TRADING_DAYS_PER_YEAR = 252

# --- Backtesting defaults ---
DEFAULT_TRANSACTION_COST = 0.001  # 10 bps per unit of turnover
DEFAULT_INITIAL_TRAIN_YEARS = 10
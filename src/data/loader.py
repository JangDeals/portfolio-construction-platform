"""
Market data acquisition module.
Downloads, validates, and saves historical price and volume data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

# Add near the top of src/data/loader.py
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from src.utils.paths import RAW_DATA_DIR
from src.config import TICKERS, START_DATE, END_DATE


def download_asset_data(tickers: list[str], start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download historical daily price and volume data for a list of tickers
    from Yahoo Finance.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols to download.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (close_prices, volumes) — both with dates as index, tickers as columns.

    Raises
    ------
    ValueError
        If any requested ticker returned no data.
    """
    logger.info(f"Downloading data for {len(tickers)} tickers: {tickers}")
    logger.info(f"Date range: {start_date} to {end_date}")

    raw_data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        group_by="ticker",
        progress=True,
    )

    close_prices = pd.DataFrame({
        ticker: raw_data[ticker]["Close"]
        for ticker in tickers
    })

    volumes = pd.DataFrame({
        ticker: raw_data[ticker]["Volume"]
        for ticker in tickers
    })

    close_prices = close_prices.dropna(how="any")
    volumes = volumes.loc[close_prices.index]

    missing_tickers = [t for t in tickers if close_prices[t].isna().all()]
    if missing_tickers:
        logger.error(f"No data returned for tickers: {missing_tickers}")
        raise ValueError(f"No data returned for tickers: {missing_tickers}")

    logger.info(f"Successfully downloaded data: {close_prices.shape[0]} rows, {close_prices.shape[1]} tickers")
    logger.info(f"Date range returned: {close_prices.index.min().date()} to {close_prices.index.max().date()}")

    return close_prices, volumes


def save_raw_data(prices: pd.DataFrame, volumes: pd.DataFrame, tickers: list[str], start_date: str, end_date: str) -> None:
    """
    Save downloaded price and volume data to data/raw/ along with a metadata
    file recording exactly what was downloaded and when.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    prices_path = RAW_DATA_DIR / "asset_prices.csv"
    prices.to_csv(prices_path)

    volumes_path = RAW_DATA_DIR / "asset_volumes.csv"
    volumes.to_csv(volumes_path)

    metadata = {
        "download_timestamp": datetime.now().isoformat(),
        "tickers": tickers,
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "actual_start_date": str(prices.index.min().date()),
        "actual_end_date": str(prices.index.max().date()),
        "num_rows": len(prices),
        "num_tickers": len(tickers),
    }
    metadata_path = RAW_DATA_DIR / "asset_prices_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Saved price data to: {prices_path}")
    logger.info(f"Saved volume data to: {volumes_path}")
    logger.info(f"Saved metadata to: {metadata_path}")
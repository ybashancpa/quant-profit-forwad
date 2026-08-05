"""
Data Loader Module
==================
Downloads and caches ETF price data using yfinance.
Uses Adjusted Close prices (dividend and split adjusted).
"""

import os
import pandas as pd
import yfinance as yf
from config import TICKERS, DATA_START, DATA_END, DATA_DIR, PRICE_FILE


def download_prices(tickers=None, start=None, end=None, force_refresh=False):
    """
    Download adjusted close prices for the ETF universe.
    Caches to CSV to avoid repeated API calls.
    
    Returns:
        pd.DataFrame: Daily adjusted close prices, columns = tickers
    """
    if tickers is None:
        tickers = TICKERS
    if start is None:
        start = DATA_START
    if end is None:
        end = DATA_END

    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, PRICE_FILE)

    # Use cache if available and not forcing refresh
    if os.path.exists(cache_path) and not force_refresh:
        print(f"[DataLoader] Loading cached data from {cache_path}")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # Filter to requested tickers
        available = [t for t in tickers if t in prices.columns]
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            print(f"[DataLoader] WARNING: Missing tickers in cache: {missing}")
        return prices[available]

    print(f"[DataLoader] Downloading data for {len(tickers)} ETFs: {tickers}")
    print(f"[DataLoader] Period: {start} to {end or 'today'}")

    # Download using yfinance
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,  # Adjusted close (dividends + splits)
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns when multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    # Clean data
    prices = prices.dropna(how="all")  # Drop rows where all are NaN
    
    # Report missing data
    print(f"\n[DataLoader] Data summary:")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Trading days: {len(prices)}")
    print(f"  Tickers: {list(prices.columns)}")
    
    # Check for NaNs per ticker
    nan_counts = prices.isna().sum()
    if nan_counts.any():
        print(f"\n  WARNING - Missing values per ticker:")
        for ticker, count in nan_counts[nan_counts > 0].items():
            print(f"    {ticker}: {count} missing days")
    
    # Save cache
    prices.to_csv(cache_path)
    print(f"\n[DataLoader] Saved to {cache_path}")

    return prices


def get_returns(prices, freq="D"):
    """Calculate simple returns from prices."""
    return prices.pct_change()


def get_log_returns(prices):
    """Calculate log returns from prices."""
    import numpy as np
    return np.log(prices / prices.shift(1))


if __name__ == "__main__":
    prices = download_prices()
    print("\nFirst 5 rows:")
    print(prices.head())
    print("\nLast 5 rows:")
    print(prices.tail())
"""
data_collector.py
-----------------
Sprint 1: Download 5 years of daily OHLCV data for Indian stocks.
Saves each stock as a CSV inside the ../data/ folder.

Usage:
    python src/data_collector.py
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

STOCKS = [
    "RELIANCE.NS",
    "INFY.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
]

YEARS = 5
INTERVAL = "1d"                          # daily candles
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
RETRY_LIMIT = 3
RETRY_DELAY = 5   # seconds between retries

# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def create_data_dir() -> None:
    """Create data/ directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Data directory ready: {DATA_DIR}")


def download_stock(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Download OHLCV data for a single stock with retry logic.
    Returns a cleaned DataFrame or None on failure.
    """
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            log.info(f"Downloading {symbol}  (attempt {attempt}/{RETRY_LIMIT})")
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=INTERVAL,
                auto_adjust=True,   # adjusts for splits/dividends
                progress=False,
            )

            if df.empty:
                log.warning(f"{symbol}: empty response from yfinance.")
                raise ValueError("Empty DataFrame")

            # yfinance sometimes returns MultiIndex columns — flatten them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Keep only OHLCV columns
            available = REQUIRED_COLUMNS & set(df.columns)
            if available != REQUIRED_COLUMNS:
                missing = REQUIRED_COLUMNS - available
                raise ValueError(f"Missing columns: {missing}")

            df = df[list(REQUIRED_COLUMNS)].copy()
            df.index.name = "Date"
            df.sort_index(inplace=True)
            df.dropna(inplace=True)

            log.info(f"{symbol}: {len(df)} rows downloaded  "
                     f"({df.index[0].date()} → {df.index[-1].date()})")
            return df

        except Exception as exc:
            log.error(f"{symbol}: attempt {attempt} failed — {exc}")
            if attempt < RETRY_LIMIT:
                log.info(f"Retrying in {RETRY_DELAY}s …")
                time.sleep(RETRY_DELAY)

    log.error(f"{symbol}: all {RETRY_LIMIT} attempts failed. Skipping.")
    return None


def save_csv(df: pd.DataFrame, symbol: str) -> Path:
    """Save DataFrame to data/<TICKER>.csv (ticker without .NS suffix)."""
    ticker = symbol.replace(".NS", "").replace(".BO", "")
    path = DATA_DIR / f"{ticker}.csv"
    df.to_csv(path)
    log.info(f"Saved → {path}  ({os.path.getsize(path) / 1024:.1f} KB)")
    return path


def verify_csv(path: Path) -> bool:
    """Re-read saved CSV and confirm all required columns are present."""
    try:
        check = pd.read_csv(path, index_col="Date", parse_dates=True)
        missing = REQUIRED_COLUMNS - set(check.columns)
        if missing:
            log.error(f"Verification FAILED for {path.name}: missing {missing}")
            return False
        log.info(f"Verification OK → {path.name}  "
                 f"rows={len(check)}, cols={list(check.columns)}")
        return True
    except Exception as exc:
        log.error(f"Could not verify {path.name}: {exc}")
        return False

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("AI-Trader  |  Sprint 1  |  Data Collector")
    log.info("=" * 60)

    create_data_dir()

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=YEARS * 365)).strftime("%Y-%m-%d")
    log.info(f"Date range: {start_date} → {end_date}")

    results = {"success": [], "failed": []}

    for symbol in STOCKS:
        df = download_stock(symbol, start_date, end_date)
        if df is not None:
            path = save_csv(df, symbol)
            ok   = verify_csv(path)
            if ok:
                results["success"].append(symbol)
            else:
                results["failed"].append(symbol)
        else:
            results["failed"].append(symbol)
        time.sleep(1)   # be polite to the API

    # ── Summary ──────────────────────────────
    log.info("=" * 60)
    log.info(f"DONE  ✅  Success: {len(results['success'])}  "
             f"❌  Failed: {len(results['failed'])}")
    if results["success"]:
        log.info(f"  Saved: {', '.join(results['success'])}")
    if results["failed"]:
        log.warning(f"  Failed: {', '.join(results['failed'])}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

"""
feature_engineer.py  (v3 — Sprint 3.5 feature upgrade)
--------------------------------------------------------
Adds 15 new predictive features on top of the existing indicators.

New features:
  Group 1 - Momentum   : Return_1D, Return_3D, Return_5D, Return_10D
  Group 2 - Volume     : Volume_MA20, Volume_Ratio, Volume_Spike
  Group 3 - Trend      : EMA20_Slope, EMA50_Slope
  Group 4 - Volatility : ATR_14, ATR_Percent
  Group 5 - Position   : Distance_EMA20, Distance_EMA50,
                         Distance_BB_High, Distance_BB_Low

Target remains leak-free:
  Signal = 1 if Close[t+5] > Close[t] * 1.02 else 0
  Last 5 rows are always dropped.

Usage:
    python src/feature_engineer.py
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────

TICKERS     = ["RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK"]
DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
LOOKAHEAD   = 5
TARGET_RISE = 0.02

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Base indicators (unchanged from v2) ──────────────────────────

def add_rsi(close, window=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_macd(close, fast=12, slow=26, signal=9):
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def add_bollinger(close, window=20, num_std=2):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return mid + num_std * std, mid, mid - num_std * std

def add_ema(close, span):
    return close.ewm(span=span, adjust=False).mean()

def add_sma(close, window):
    return close.rolling(window).mean()

def add_vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()

# ── New features ─────────────────────────────────────────────────

def add_momentum_returns(df):
    """Group 1: multi-period returns — how strongly is price moving?"""
    close = df["Close"]
    df["Return_1D"]  = close.pct_change(1)
    df["Return_3D"]  = close.pct_change(3)
    df["Return_5D"]  = close.pct_change(5)
    df["Return_10D"] = close.pct_change(10)
    return df

def add_volume_intelligence(df):
    """Group 2: volume tells us conviction behind price moves."""
    vol = df["Volume"]
    df["Volume_MA20"]  = vol.rolling(20).mean()
    df["Volume_Ratio"] = vol / df["Volume_MA20"].replace(0, np.nan)
    # Spike = 1 when today's volume is >1.5x the 20-day average
    df["Volume_Spike"] = (df["Volume_Ratio"] > 1.5).astype(int)
    return df

def add_trend_strength(df):
    """Group 3: slope of EMAs — is the trend accelerating or slowing?"""
    df["EMA20_Slope"] = df["EMA_20"].diff(5)
    df["EMA50_Slope"] = df["EMA_50"].diff(5)
    return df

def add_atr(df, window=14):
    """Group 4: Average True Range — raw and % of price."""
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR_14"]      = tr.rolling(window).mean()
    df["ATR_Percent"] = df["ATR_14"] / close   # normalised — comparable across stocks
    return df

def add_relative_position(df):
    """Group 5: how far is price from key levels? (normalised)"""
    close = df["Close"]
    df["Distance_EMA20"]   = (close - df["EMA_20"])   / df["EMA_20"]
    df["Distance_EMA50"]   = (close - df["EMA_50"])   / df["EMA_50"]
    df["Distance_BB_High"] = (close - df["BB_High"])  / df["BB_High"]
    df["Distance_BB_Low"]  = (close - df["BB_Low"])   / df["BB_Low"]
    return df

# ── Full pipeline ─────────────────────────────────────────────────

def build_features(df):
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # ── Base indicators ───────────────────────────────────────────
    df["RSI_14"] = add_rsi(close)

    df["MACD"], df["MACD_Signal"], df["MACD_Diff"] = add_macd(close)

    df["BB_High"], df["BB_Mid"], df["BB_Low"] = add_bollinger(close)

    df["SMA_20"] = add_sma(close, 20)
    df["SMA_50"] = add_sma(close, 50)
    df["EMA_20"] = add_ema(close, 20)   # needed by slope + distance
    df["EMA_50"] = add_ema(close, 50)   # needed by slope + distance
    df["EMA_12"] = add_ema(close, 12)
    df["EMA_26"] = add_ema(close, 26)

    df["VWAP"]        = add_vwap(df)
    df["Returns"]     = close.pct_change()
    df["Volatility"]  = df["Returns"].rolling(14).std()
    df["Price_Range"] = (high - low) / close

    # ── New features (Sprint 3.5) ─────────────────────────────────
    df = add_momentum_returns(df)       # Return_1D … Return_10D
    df = add_volume_intelligence(df)    # Volume_MA20, Volume_Ratio, Volume_Spike
    df = add_trend_strength(df)         # EMA20_Slope, EMA50_Slope
    df = add_atr(df)                    # ATR_14, ATR_Percent
    df = add_relative_position(df)      # Distance_EMA20/50/BB_High/Low

    # ── Leak-free target ─────────────────────────────────────────
    future_close = close.shift(-LOOKAHEAD)
    df["Signal"] = (future_close > close * (1 + TARGET_RISE)).astype("Int64")

    return df

# ── Per-ticker processing ─────────────────────────────────────────

def process_ticker(ticker):
    raw_path = DATA_DIR / f"{ticker}.csv"
    if not raw_path.exists():
        log.error(f"{ticker}: raw CSV not found → {raw_path}"); return False

    try:
        df = pd.read_csv(raw_path, index_col="Date", parse_dates=True)
        log.info(f"{ticker}: loaded {len(df)} rows")

        df = build_features(df)

        before = len(df)
        df = df.iloc[:-LOOKAHEAD]   # drop last 5 — future unknown
        df.dropna(inplace=True)     # drop rolling warmup rows
        after  = len(df)
        log.info(f"{ticker}: kept {after} rows (dropped {before - after})")

        # Feature count
        non_feat = {"Open","High","Low","Close","Volume","Signal"}
        feat_cols = [c for c in df.columns if c not in non_feat]
        log.info(f"{ticker}: {len(feat_cols)} feature columns")

        # Class balance
        buy_pct = df["Signal"].mean() * 100
        log.info(f"{ticker}: BUY={buy_pct:.1f}%  HOLD/SELL={100-buy_pct:.1f}%")

        out_path = DATA_DIR / f"{ticker}_features.csv"
        df.to_csv(out_path)
        log.info(f"{ticker}: saved → {out_path.name}  "
                 f"({out_path.stat().st_size/1024:.1f} KB)  ✅")
        return True

    except Exception as exc:
        log.error(f"{ticker}: FAILED — {exc}")
        import traceback; traceback.print_exc()
        return False

# ── Main ──────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("AI-Trader  |  Sprint 3.5  |  Feature Upgrade (30 features)")
    log.info("=" * 60)

    results = {"success": [], "failed": []}
    for ticker in TICKERS:
        ok = process_ticker(ticker)
        (results["success"] if ok else results["failed"]).append(ticker)

    log.info("=" * 60)
    log.info(f"DONE  ✅ Success: {len(results['success'])}  "
             f"❌ Failed: {len(results['failed'])}")
    log.info("")
    log.info("New features added (15):")
    log.info("  Momentum : Return_1D, Return_3D, Return_5D, Return_10D")
    log.info("  Volume   : Volume_MA20, Volume_Ratio, Volume_Spike")
    log.info("  Trend    : EMA20_Slope, EMA50_Slope")
    log.info("  Volatility: ATR_14, ATR_Percent")
    log.info("  Position : Distance_EMA20, Distance_EMA50,")
    log.info("             Distance_BB_High, Distance_BB_Low")
    log.info("")
    log.info("Next: python src/train_model.py")
    log.info("Target: CV F1 > 0.40  |  Holdout F1 > 0.35")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
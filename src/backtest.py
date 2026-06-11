"""
backtest.py
-----------
Sprint 4: Walk-forward backtesting engine.

Rules (Production v1):
  - Entry : BUY signal → enter at NEXT day's Open
  - Exit  : Close of day 3 after entry
  - Size  : 10% of current capital per trade
  - No stop-loss, no take-profit (test pure model signal first)

Usage:
    python src/backtest.py
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# ── Config ────────────────────────────────────────────────────────

TICKERS     = ["TCS", "HDFCBANK"]          # best two models first
DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR  = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

LOOKAHEAD       = 3
TARGET_RISE     = 0.0          # Target C: any positive move
POSITION_SIZE   = 0.10         # 10% of capital per trade
INITIAL_CAPITAL = 100_000.0    # ₹1,00,000 (normalised — scales to any amount)
RISK_FREE_RATE  = 0.065        # ~6.5% annual (Indian 10yr bond approx)

NON_FEATURE = {"Open", "High", "Low", "Close", "Volume", "Signal", "Target", "_target"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Data helpers ──────────────────────────────────────────────────

def load_data(ticker):
    path = DATA_DIR / f"{ticker}_features.csv"
    if not path.exists():
        log.error(f"{ticker}: features CSV not found"); return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df


def build_target_c(df):
    """Rebuild Target C leak-free (same as training)."""
    close    = df["Close"]
    future   = close.shift(-LOOKAHEAD)
    df       = df.copy()
    df["_y"] = (future > close * (1 + TARGET_RISE)).astype("Int64")
    df       = df.iloc[:-LOOKAHEAD]
    df.dropna(inplace=True)
    return df


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in NON_FEATURE | {"_y"}
            and pd.api.types.is_numeric_dtype(df[c])]


def load_model(ticker):
    path = MODELS_DIR / f"{ticker}_model.json"
    if not path.exists():
        log.error(f"{ticker}: model not found at {path}"); return None
    model = XGBClassifier()
    model.load_model(str(path))
    log.info(f"{ticker}: model loaded from {path.name}")
    return model

# ── Backtesting engine ────────────────────────────────────────────

def run_backtest(ticker):
    df = load_data(ticker)
    if df is None: return None

    df = build_target_c(df)
    feat_cols = get_feature_cols(df)

    model = load_model(ticker)
    if model is None: return None

    X = df[feat_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    # ── Use only the last 20% of data as the backtest window ──────
    # (same split used in training — model has never seen this data)
    split        = int(len(df) * 0.80)
    df_test      = df.iloc[split:].copy()
    X_test       = X.iloc[split:].copy()

    log.info(f"{ticker}: backtesting on {len(df_test)} days "
             f"({df_test.index[0].date()} → {df_test.index[-1].date()})")

    # ── Walk-forward simulation ───────────────────────────────────
    predictions  = model.predict(X_test)
    closes       = df_test["Close"].values
    opens        = df_test["Open"].values
    dates        = df_test.index

    capital      = INITIAL_CAPITAL
    trades       = []
    active_trade = None   # dict when a trade is open

    for i in range(len(df_test)):
        # ── Check if an active trade should exit ──
        if active_trade is not None:
            days_held = i - active_trade["entry_bar"]
            if days_held >= LOOKAHEAD:
                exit_price  = closes[i]
                entry_price = active_trade["entry_price"]
                trade_ret   = (exit_price - entry_price) / entry_price
                pnl         = active_trade["size"] * trade_ret
                capital    += pnl

                trades.append({
                    "entry_date" : active_trade["entry_date"],
                    "exit_date"  : dates[i],
                    "entry_price": round(entry_price, 2),
                    "exit_price" : round(exit_price,  2),
                    "return_pct" : round(trade_ret * 100, 4),
                    "pnl"        : round(pnl, 2),
                    "capital"    : round(capital, 2),
                    "win"        : int(trade_ret > 0),
                })
                active_trade = None

        # ── Check for new BUY signal (only if no open trade) ──────
        # Enter at NEXT bar's open (i+1), so we need i+1 to exist
        if active_trade is None and predictions[i] == 1:
            if i + 1 < len(df_test):
                entry_price  = opens[i + 1]
                trade_size   = capital * POSITION_SIZE
                active_trade = {
                    "entry_bar"  : i + 1,
                    "entry_date" : dates[i + 1],
                    "entry_price": entry_price,
                    "size"       : trade_size,
                }

    # Close any trade still open at end of backtest
    if active_trade is not None:
        exit_price  = closes[-1]
        entry_price = active_trade["entry_price"]
        trade_ret   = (exit_price - entry_price) / entry_price
        pnl         = active_trade["size"] * trade_ret
        capital    += pnl
        trades.append({
            "entry_date" : active_trade["entry_date"],
            "exit_date"  : dates[-1],
            "entry_price": round(entry_price, 2),
            "exit_price" : round(exit_price, 2),
            "return_pct" : round(trade_ret * 100, 4),
            "pnl"        : round(pnl, 2),
            "capital"    : round(capital, 2),
            "win"        : int(trade_ret > 0),
        })

    if not trades:
        log.warning(f"{ticker}: no trades generated"); return None

    trades_df = pd.DataFrame(trades)

    # ── Metrics ───────────────────────────────────────────────────
    n_trades      = len(trades_df)
    win_rate      = trades_df["win"].mean() * 100
    total_return  = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    avg_trade_ret = trades_df["return_pct"].mean()
    avg_win       = trades_df.loc[trades_df["win"]==1, "return_pct"].mean() if trades_df["win"].sum() > 0 else 0
    avg_loss      = trades_df.loc[trades_df["win"]==0, "return_pct"].mean() if (trades_df["win"]==0).sum() > 0 else 0

    # Max drawdown from equity curve
    equity        = trades_df["capital"].values
    peak          = np.maximum.accumulate(equity)
    drawdown      = (equity - peak) / peak * 100
    max_drawdown  = drawdown.min()

    # Sharpe ratio (annualised, assuming ~250 trading days/year)
    # Using per-trade returns, scaled to daily
    daily_ret     = trades_df["return_pct"].values / 100
    trading_days  = (trades_df["exit_date"].iloc[-1] -
                     trades_df["entry_date"].iloc[0]).days
    trades_per_yr = n_trades / max(trading_days / 365, 0.1)
    rf_per_trade  = RISK_FREE_RATE / max(trades_per_yr, 1)
    excess        = daily_ret - rf_per_trade
    sharpe        = (excess.mean() / excess.std() * np.sqrt(trades_per_yr)
                     if excess.std() > 0 else 0.0)

    # Profit factor
    gross_win  = trades_df.loc[trades_df["win"]==1, "pnl"].sum()
    gross_loss = abs(trades_df.loc[trades_df["win"]==0, "pnl"].sum())
    profit_fac = gross_win / gross_loss if gross_loss > 0 else float("inf")

    result = dict(
        ticker         = ticker,
        period         = f"{df_test.index[0].date()} → {df_test.index[-1].date()}",
        n_trades       = n_trades,
        win_rate       = round(win_rate, 1),
        total_return   = round(total_return, 2),
        avg_trade_pct  = round(avg_trade_ret, 3),
        avg_win_pct    = round(avg_win, 3),
        avg_loss_pct   = round(avg_loss, 3),
        profit_factor  = round(profit_fac, 3),
        max_drawdown   = round(max_drawdown, 2),
        sharpe_ratio   = round(sharpe, 3),
        final_capital  = round(capital, 2),
    )

    # ── Log per-ticker detail ─────────────────────────────────────
    log.info(f"\n{'─'*50}")
    log.info(f"{ticker} BACKTEST RESULTS")
    log.info(f"{'─'*50}")
    log.info(f"  Period         : {result['period']}")
    log.info(f"  Trades         : {n_trades}")
    log.info(f"  Win Rate       : {win_rate:.1f}%")
    log.info(f"  Total Return   : {total_return:+.2f}%")
    log.info(f"  Avg Trade      : {avg_trade_ret:+.3f}%")
    log.info(f"  Avg Win        : {avg_win:+.3f}%")
    log.info(f"  Avg Loss       : {avg_loss:+.3f}%")
    log.info(f"  Profit Factor  : {profit_fac:.3f}")
    log.info(f"  Max Drawdown   : {max_drawdown:.2f}%")
    log.info(f"  Sharpe Ratio   : {sharpe:.3f}")
    log.info(f"  Final Capital  : ₹{capital:,.2f}  "
             f"(started ₹{INITIAL_CAPITAL:,.0f})")

    verdict = "✅ POSITIVE EXPECTANCY" if total_return > 0 and profit_fac > 1 else "❌ NEGATIVE EXPECTANCY"
    log.info(f"  Verdict        : {verdict}")

    # Save trade log
    trade_log_path = RESULTS_DIR / f"{ticker}_trade_log.csv"
    trades_df.to_csv(trade_log_path, index=False)
    log.info(f"  Trade log      : {trade_log_path.name}")

    return result

# ── Main ──────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("AI-Trader  |  Sprint 4  |  Backtesting Engine")
    log.info(f"  Target  : Price higher after {LOOKAHEAD} days (Target C)")
    log.info(f"  Size    : {POSITION_SIZE*100:.0f}% of capital per trade")
    log.info(f"  Capital : ₹{INITIAL_CAPITAL:,.0f} (starting)")
    log.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for ticker in TICKERS:
        r = run_backtest(ticker)
        if r:
            all_results.append(r)

    if not all_results:
        log.error("No backtest results."); return

    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(RESULTS_DIR / "backtest_summary.csv", index=False)

    # ── Final summary table ───────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("BACKTEST SUMMARY")
    log.info(f"{'Ticker':<11} {'Trades':>7} {'WinRate':>8} {'Return%':>8} "
             f"{'AvgTrd%':>8} {'MaxDD%':>8} {'Sharpe':>7} {'PF':>6}")
    log.info("─" * 60)
    for r in all_results:
        log.info(
            f"{r['ticker']:<11} "
            f"{r['n_trades']:>7} "
            f"{r['win_rate']:>7.1f}% "
            f"{r['total_return']:>+7.2f}% "
            f"{r['avg_trade_pct']:>+7.3f}% "
            f"{r['max_drawdown']:>7.2f}% "
            f"{r['sharpe_ratio']:>7.3f} "
            f"{r['profit_factor']:>6.3f}"
        )
    log.info("=" * 60)
    log.info("Metrics guide:")
    log.info("  Total Return > 0%   → model made money")
    log.info("  Profit Factor > 1.0 → wins outweigh losses")
    log.info("  Sharpe > 1.0        → good risk-adjusted return")
    log.info("  Max Drawdown < -20% → risky, needs stop-loss")


if __name__ == "__main__":
    main()
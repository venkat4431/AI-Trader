"""
target_discovery.py
--------------------
Target Discovery Sprint: Tests 4 target definitions across 5 stocks.
20 experiments total. Finds which target contains a learnable signal.

Targets tested:
  A: Close[t+5] > Close[t] * 1.02  (2% rise in 5 days — current)
  B: Close[t+5] > Close[t] * 1.01  (1% rise in 5 days)
  C: Close[t+3] > Close[t]         (any rise in 3 days)
  D: Close[t+10] > Close[t] * 1.03 (3% rise in 10 days)

Output: results/target_discovery.csv  (sorted by Holdout F1)

Usage:
    python src/target_discovery.py
"""

import logging
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Config ────────────────────────────────────────────────────────

TICKERS = ["RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK"]

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# 4 target definitions: (lookahead_days, min_rise_pct, label)
TARGETS = [
    (5,  0.02, "A: 2% in 5d"),
    (5,  0.01, "B: 1% in 5d"),
    (3,  0.00, "C: any in 3d"),
    (10, 0.03, "D: 3% in 10d"),
]

TEST_SIZE   = 0.20
N_CV_SPLITS = 5
NON_FEATURE = {"Open", "High", "Low", "Close", "Volume", "Signal", "Target"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Load raw features (no target column — we build it fresh each run) ──

def load_base(ticker):
    """Load features CSV, drop any existing target columns."""
    path = DATA_DIR / f"{ticker}_features.csv"
    if not path.exists():
        log.error(f"{ticker}: features CSV not found"); return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.drop(columns=[c for c in ("Signal", "Target") if c in df.columns],
            inplace=True)
    return df

# ── Build target for a given (lookahead, rise) ────────────────────

def build_target(df, lookahead, rise):
    close  = df["Close"]
    future = close.shift(-lookahead)
    y      = (future > close * (1 + rise)).astype("Int64")
    df     = df.copy()
    df["_target"] = y
    # Drop last `lookahead` rows (future unknown) + warmup NaNs
    df = df.iloc[:-lookahead]
    df.dropna(inplace=True)
    return df

# ── Feature columns ───────────────────────────────────────────────

def get_features(df):
    return [c for c in df.columns
            if c not in NON_FEATURE | {"_target"}
            and pd.api.types.is_numeric_dtype(df[c])]

# ── Model ─────────────────────────────────────────────────────────

def make_model(spw):
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )

# ── Single experiment ─────────────────────────────────────────────

def run_experiment(ticker, lookahead, rise, label):
    base = load_base(ticker)
    if base is None:
        return None

    df = build_target(base, lookahead, rise)
    if len(df) < 200:
        log.warning(f"{ticker} | {label}: too few rows ({len(df)}), skipping")
        return None

    feat_cols = get_features(df)
    X = df[feat_cols].copy()
    y = df["_target"].astype(int)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    buy_pct = n_pos / len(y) * 100

    if n_pos < 10:
        log.warning(f"{ticker} | {label}: only {n_pos} positive samples, skipping")
        return None

    spw   = float(n_neg / n_pos)
    split = int(len(X) * (1 - TEST_SIZE))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    # Walk-forward CV
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_f1, cv_prec = [], []
    for tr_i, val_i in tscv.split(X_tr):
        m = make_model(spw)
        m.fit(X_tr.iloc[tr_i], y_tr.iloc[tr_i], verbose=False)
        p = m.predict(X_tr.iloc[val_i])
        cv_f1.append(f1_score(y_tr.iloc[val_i], p, zero_division=0))
        cv_prec.append(precision_score(y_tr.iloc[val_i], p, zero_division=0))

    # Final model on full train set
    final = make_model(spw)
    final.fit(X_tr, y_tr, verbose=False)
    preds = final.predict(X_te)

    result = dict(
        ticker    = ticker,
        target    = label,
        lookahead = lookahead,
        rise_pct  = f"{rise*100:.0f}%",
        buy_rate  = f"{buy_pct:.1f}%",
        cv_f1     = round(float(np.mean(cv_f1)),  4),
        cv_prec   = round(float(np.mean(cv_prec)), 4),
        hold_acc  = round(accuracy_score(y_te, preds), 4),
        hold_prec = round(precision_score(y_te, preds, zero_division=0), 4),
        hold_rec  = round(recall_score(y_te, preds,    zero_division=0), 4),
        hold_f1   = round(f1_score(y_te, preds,        zero_division=0), 4),
    )

    log.info(
        f"  {ticker:<10} {label:<15} "
        f"buy={buy_pct:4.1f}%  "
        f"CV_F1={result['cv_f1']:.3f}  "
        f"Hold_F1={result['hold_f1']:.3f}  "
        f"Hold_Prec={result['hold_prec']:.3f}"
    )
    return result

# ── Main ──────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("AI-Trader  |  Target Discovery Sprint  |  20 Experiments")
    log.info("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for (lookahead, rise, label), ticker in product(TARGETS, TICKERS):
        r = run_experiment(ticker, lookahead, rise, label)
        if r:
            all_results.append(r)

    if not all_results:
        log.error("No experiments completed."); return

    df_res = pd.DataFrame(all_results)
    df_res.sort_values("hold_f1", ascending=False, inplace=True)
    df_res.reset_index(drop=True, inplace=True)

    out_path = RESULTS_DIR / "target_discovery.csv"
    df_res.to_csv(out_path, index=False)

    # ── Summary table ─────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("RESULTS  (sorted by Holdout F1, best first)")
    log.info(f"{'#':<3} {'Ticker':<11} {'Target':<16} {'BUY%':>5} "
             f"{'CV_F1':>6} {'H_Prec':>7} {'H_Rec':>6} {'H_F1':>6}")
    log.info("─" * 70)
    for i, row in df_res.iterrows():
        flag = " ✅" if row["hold_f1"] >= 0.35 and row["hold_prec"] >= 0.40 else ""
        log.info(
            f"{i+1:<3} {row['ticker']:<11} {row['target']:<16} "
            f"{row['buy_rate']:>5} "
            f"{row['cv_f1']:>6.3f} "
            f"{row['hold_prec']:>7.3f} "
            f"{row['hold_rec']:>6.3f} "
            f"{row['hold_f1']:>6.3f}"
            f"{flag}"
        )

    # ── Best target per stock ─────────────────────────────────────
    log.info("\n" + "─" * 70)
    log.info("BEST TARGET PER STOCK:")
    for ticker in TICKERS:
        sub = df_res[df_res["ticker"] == ticker]
        if sub.empty: continue
        best = sub.iloc[0]
        log.info(f"  {ticker:<11} → {best['target']:<16} "
                 f"Hold_F1={best['hold_f1']:.3f}  Prec={best['hold_prec']:.3f}")

    # ── Winner ────────────────────────────────────────────────────
    passed = df_res[(df_res["hold_f1"] >= 0.35) & (df_res["hold_prec"] >= 0.40)]
    log.info("\n" + "─" * 70)
    if not passed.empty:
        log.info(f"✅ {len(passed)} experiment(s) passed threshold "
                 f"(Hold F1≥0.35 AND Prec≥0.40)")
        winner = passed.iloc[0]
        log.info(f"   Best overall: {winner['ticker']} with {winner['target']}")
        log.info(f"   → LOOKAHEAD={winner['lookahead']}  "
                 f"RISE={winner['rise_pct']}  F1={winner['hold_f1']}")
        log.info("   Recommendation: USE THIS TARGET → proceed to Sprint 4")
    else:
        log.info("❌ No experiment passed both thresholds (F1≥0.35 AND Prec≥0.40)")
        log.info("   Recommendation: Target redesign or expand to more stocks/data")

    log.info("=" * 70)
    log.info(f"Full results → {out_path}")

if __name__ == "__main__":
    main()
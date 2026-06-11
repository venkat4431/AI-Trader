"""
train_model.py  (v3 — class-imbalance fix)
------------------------------------------
Sprint 3: Adds scale_pos_weight to handle the ~25% BUY / 75% HOLD
class imbalance. Without this, XGBoost predicts HOLD_SELL almost
always and achieves fake-high accuracy with near-zero F1.

Usage:
    python src/train_model.py
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

# ── Config ────────────────────────────────────────────────────────

TICKERS     = ["RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK"]
DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR  = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

NON_FEATURE_COLS = {"Date", "Open", "High", "Low", "Close", "Volume",
                    "Target", "Signal", "Future"}
TARGET_COL  = "Target"
SIGNAL_COL  = "Signal"
TEST_SIZE   = 0.20
N_CV_SPLITS = 5

# ── Logging ───────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────

def load_features(ticker):
    path = DATA_DIR / f"{ticker}_features.csv"
    if not path.exists():
        log.error(f"{ticker}: not found → {path}"); return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    log.info(f"{ticker}: loaded {len(df)} rows")
    return df


def resolve_target(df, ticker):
    if TARGET_COL in df.columns: return TARGET_COL
    if SIGNAL_COL in df.columns:
        log.info(f"{ticker}: using 'Signal' as target"); return SIGNAL_COL
    log.error(f"{ticker}: no target column found"); return None


def get_feature_cols(df, target_col):
    skip = NON_FEATURE_COLS | {target_col}
    return [c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def compute_scale_pos_weight(y_train):
    """
    scale_pos_weight = count(negative) / count(positive)
    Tells XGBoost to penalise missing BUY signals more heavily.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    ratio = n_neg / n_pos if n_pos > 0 else 1.0
    log.info(f"  class ratio (neg/pos) = {ratio:.2f}  "
             f"[neg={n_neg}, pos={n_pos}]")
    return float(ratio)


def make_model(scale_pos_weight):
    return XGBClassifier(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos_weight,   # ← key fix
        eval_metric       = "logloss",
        random_state      = 42,
        n_jobs            = -1,
    )


def time_series_cv(X, y, spw, ticker):
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for fold, (tr, val) in enumerate(tscv.split(X), 1):
        m = make_model(spw)
        m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        p = m.predict(X.iloc[val])
        metrics["accuracy"].append(accuracy_score(y.iloc[val], p))
        metrics["precision"].append(precision_score(y.iloc[val], p, zero_division=0))
        metrics["recall"].append(recall_score(y.iloc[val], p, zero_division=0))
        metrics["f1"].append(f1_score(y.iloc[val], p, zero_division=0))
        log.info(f"  Fold {fold}: acc={metrics['accuracy'][-1]:.3f}  "
                 f"prec={metrics['precision'][-1]:.3f}  "
                 f"f1={metrics['f1'][-1]:.3f}")
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def evaluate_holdout(model, X_test, y_test, ticker):
    preds  = model.predict(X_test)
    proba  = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, preds)
    prec   = precision_score(y_test, preds, zero_division=0)
    rec    = recall_score(y_test, preds, zero_division=0)
    f1     = f1_score(y_test, preds, zero_division=0)
    cm     = confusion_matrix(y_test, preds)
    last_p = float(proba[-1])
    signal = "BUY" if last_p >= 0.55 else "HOLD_SELL"
    log.info(f"{ticker} HOLDOUT → acc={acc:.3f}  prec={prec:.3f}  "
             f"rec={rec:.3f}  f1={f1:.3f}")
    log.info(f"  Confusion: TN={cm[0,0]} FP={cm[0,1]} | "
             f"FN={cm[1,0]} TP={cm[1,1]}")
    log.info(f"{ticker} TODAY  → confidence={last_p:.1%}  signal={signal}")
    return dict(
        ticker=ticker,
        holdout_accuracy=round(acc,4),  holdout_precision=round(prec,4),
        holdout_recall=round(rec,4),    holdout_f1=round(f1,4),
        today_confidence=round(last_p,4), today_signal=signal,
    )


def top_features(model, cols, ticker):
    ranked = sorted(zip(cols, model.feature_importances_),
                    key=lambda x: x[1], reverse=True)
    log.info(f"{ticker} top features:")
    for name, score in ranked[:5]:
        log.info(f"  {name:<25} {score:.4f}  {'█'*int(score*40)}")


def train_ticker(ticker):
    df = load_features(ticker)
    if df is None: return None

    target_col   = resolve_target(df, ticker)
    if target_col is None: return None

    feature_cols = get_feature_cols(df, target_col)
    log.info(f"{ticker}: {len(feature_cols)} features")

    if len(feature_cols) < 2:
        log.error(f"{ticker}: too few features, skipping"); return None

    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    split    = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    log.info(f"{ticker}: train={len(X_train)}  test={len(X_test)}")

    spw = compute_scale_pos_weight(y_train)

    log.info(f"{ticker}: running {N_CV_SPLITS}-fold walk-forward CV …")
    cv = time_series_cv(X_train, y_train, spw, ticker)
    log.info(f"{ticker} CV avg → acc={cv['accuracy']:.3f}  "
             f"prec={cv['precision']:.3f}  f1={cv['f1']:.3f}")

    final = make_model(spw)
    final.fit(X_train, y_train, verbose=False)

    report = evaluate_holdout(final, X_test, y_test, ticker)
    report.update({f"cv_{k}": v for k, v in cv.items()})

    top_features(final, feature_cols, ticker)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{ticker}_model.json"
    final.save_model(str(model_path))
    log.info(f"{ticker}: model saved → {model_path.name}")
    return report


# ── Main ──────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("AI-Trader  |  Sprint 3 v3  |  Model Training (imbalance fix)")
    log.info("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    reports = []
    for ticker in TICKERS:
        log.info(f"\n{'─'*40}")
        r = train_ticker(ticker)
        if r: reports.append(r)

    if not reports:
        log.error("No models trained."); return

    pd.DataFrame(reports).to_csv(RESULTS_DIR / "model_report.csv", index=False)

    log.info("\n" + "=" * 60)
    log.info("AI-Trader  |  Sprint 3 v3  |  SUMMARY")
    log.info(f"{'Ticker':<12} {'CV Acc':>7} {'CV F1':>6} {'Hold Acc':>9} "
             f"{'Prec':>7} {'Rec':>6} {'F1':>7}  Signal")
    log.info("─" * 68)
    for r in reports:
        log.info(
            f"{r['ticker']:<12} "
            f"{r['cv_accuracy']:>7.3f} "
            f"{r['cv_f1']:>6.3f} "
            f"{r['holdout_accuracy']:>9.3f} "
            f"{r['holdout_precision']:>7.3f} "
            f"{r['holdout_recall']:>6.3f} "
            f"{r['holdout_f1']:>7.3f}  "
            f"{r['today_signal']}"
        )
    log.info("=" * 68)
    log.info("Target: CV F1 > 0.35 and Holdout Precision > 0.40")
    log.info(f"Report → {RESULTS_DIR / 'model_report.csv'}")


if __name__ == "__main__":
    main()
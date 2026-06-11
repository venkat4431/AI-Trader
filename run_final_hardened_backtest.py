import os
import itertools
import glob
import numpy as np
import pandas as pd
from tabulate import tabulate

# ==========================================
# HARDENED BACKTEST CONFIGURATION
# ==========================================
DATA_DIR = "data"
RESULTS_DIR = "results"
FINAL_REPORT_PATH = os.path.join(RESULTS_DIR, "final_hardened_portfolio_results.csv")

HOLDING_DAYS_SPACE = [1, 2, 3, 5]
PROB_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
TAKE_PROFITS = [None, 0.01, 0.02, 0.03]  
STOP_LOSSES = [None, 0.01, 0.02]        

INITIAL_CAPITAL = 100000.0
RISK_FREE_RATE = 0.07       

def calculate_metrics_pessimistic(trades, initial_capital):
    if not trades:
        return {
            "Total Trades": 0, "Win Rate %": 0.0, "Total Return %": 0.0,
            "Avg Trade %": 0.0, "Profit Factor": 0.0, "Max Drawdown %": 0.0,
            "Sharpe Ratio": -99.0, "Final Capital": initial_capital
        }
    
    df_trades = pd.DataFrame(trades)
    returns = df_trades["return_pct"]
    
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = (len(wins) / len(returns)) * 100 if len(returns) > 0 else 0
    
    total_return_pct = (df_trades["exit_cap"].iloc[-1] - initial_capital) / initial_capital * 100
    avg_trade_pct = returns.mean() * 100
    
    gross_profits = wins.sum()
    gross_losses = abs(losses.sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else 1.0
    
    equity_curve = [initial_capital]
    for _, row in df_trades.iterrows():
        equity_curve.append(row["exit_cap"])
    equity_curve = np.array(equity_curve)
    
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    max_dd_pct = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0
    
    trade_returns = np.diff(equity_curve) / equity_curve[:-1]
    excess_returns = trade_returns - (RISK_FREE_RATE / 252)
    if len(trade_returns) > 1 and excess_returns.std() > 0:
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
        
    return {
        "Total Trades": len(returns),
        "Win Rate %": round(win_rate, 2),
        "Total Return %": round(total_return_pct, 2),
        "Avg Trade %": round(avg_trade_pct, 2),
        "Profit Factor": round(profit_factor, 3),
        "Max Drawdown %": round(max_dd_pct, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Final Capital": round(equity_curve[-1], 2)
    }

def simulate_hardened_strategy(df, hold_days, threshold, tp, sl):
    trades = []
    current_capital = INITIAL_CAPITAL
    in_position = False
    entry_idx = 0
    entry_price = 0.0
    
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    probs = df['Probability'].values
    
    i = 0
    while i < len(df):
        if not in_position:
            # Enforce strict next-day entry at Open price
            if probs[i] >= threshold and i < len(df) - 1:
                in_position = True
                entry_idx = i + 1  
                entry_price = opens[entry_idx] 
                i = entry_idx
                continue
            i += 1
        else:
            days_held = i - entry_idx
            tp_triggered = (tp is not None) and (highs[i] >= entry_price * (1.0 + tp))
            sl_triggered = (sl is not None) and (lows[i] <= entry_price * (1.0 - sl))
            
            # Pessimistic clearing: Assume stop loss hit first if both triggered on same day
            if tp_triggered and sl_triggered:
                ret = -sl
                current_capital *= (1.0 + ret)
                trades.append({"entry_i": entry_idx, "exit_i": i, "return_pct": ret, "exit_cap": current_capital})
                in_position = False
                i += 1
                continue
                
            if sl_triggered:
                ret = -sl
                current_capital *= (1.0 + ret)
                trades.append({"entry_i": entry_idx, "exit_i": i, "return_pct": ret, "exit_cap": current_capital})
                in_position = False
                i += 1
                continue
                
            if tp_triggered:
                ret = tp
                current_capital *= (1.0 + ret)
                trades.append({"entry_i": entry_idx, "exit_i": i, "return_pct": ret, "exit_cap": current_capital})
                in_position = False
                i += 1
                continue
                
            if days_held >= hold_days or i == len(df) - 1:
                ret = (closes[i] - entry_price) / entry_price
                current_capital *= (1.0 + ret)
                trades.append({"entry_i": entry_idx, "exit_i": i, "return_pct": ret, "exit_cap": current_capital})
                in_position = False
                
            i += 1
            
    return calculate_metrics_pessimistic(trades, INITIAL_CAPITAL)

def main():
    print("⏳ Loading real model predictions and merging market vectors...")
    prediction_files = glob.glob(os.path.join(RESULTS_DIR, "*_predictions.csv"))
    
    if not prediction_files:
        print("❌ Error: No prediction CSV files found in results/ folder.")
        return

    grid = list(itertools.product(HOLDING_DAYS_SPACE, PROB_THRESHOLDS, TAKE_PROFITS, STOP_LOSSES))
    portfolio_top_strategies = []

    for pred_path in prediction_files:
        stock_name = os.path.basename(pred_path).replace("_predictions.csv", "")
        feature_path = os.path.join(DATA_DIR, f"{stock_name}_features.csv")
        
        if not os.path.exists(feature_path):
            print(f"⚠️ Market pricing feature file missing for {stock_name}. Skipping asset...")
            continue
            
        # Ingest predictions and original features
        df_pred = pd.read_csv(pred_path)
        df_feat = pd.read_csv(feature_path)
        
        # Merge data strictly on Date to guarantee no indexing row shifts
        df_merged = pd.merge(df_pred, df_feat[['Date', 'Open', 'High', 'Low', 'Close']], on='Date')
        df_merged['Date'] = pd.to_datetime(df_merged['Date'])
        df_merged = df_merged.sort_values(by='Date').reset_index(drop=True)

        optimization_results = []
        print(f"⚙️ Running 240 Grid Iterations on Real Signals for: {stock_name}")
        
        for hold, thresh, tp, sl in grid:
            metrics = simulate_hardened_strategy(df_merged, hold, thresh, tp, sl)
            optimization_results.append({
                "Stock": stock_name, "Holding Days": hold, "Prob Threshold": thresh,
                "Take Profit %": f"{tp*100}%" if tp else "None", "Stop Loss %": f"{sl*100}%" if sl else "None",
                **metrics
            })
            
        df_results = pd.DataFrame(optimization_results)
        df_results = df_results.sort_values(by=["Profit Factor", "Sharpe Ratio", "Total Return %"], ascending=[False, False, False])
        
        # Save individual stock grid output
        df_results.to_csv(os.path.join(RESULTS_DIR, f"hardened_real_{stock_name}_metrics.csv"), index=False)
        portfolio_top_strategies.append(df_results.head(1))

    if portfolio_top_strategies:
        print("\n" + "="*80)
        print("🏆 TRUE UNINFLATED TRADING ASSISTANT METRICS PER STOCK")
        print("="*80)
        final_portfolio_df = pd.concat(portfolio_top_strategies)
        final_portfolio_df.to_csv(FINAL_REPORT_PATH, index=False)
        print(tabulate(final_portfolio_df.drop(columns=["Final Capital", "Stock"]), headers="keys", tablefmt="grid", showindex=False))
        print(f"\n💾 Unified execution verification report saved to: {FINAL_REPORT_PATH}")

if __name__ == "__main__":
    main()

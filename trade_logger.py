import os
import sys
import pandas as pd
from datetime import datetime

SIGNALS_PATH = os.path.join("results", "signals_history.csv")
PORTFOLIO_PATH = os.path.join("results", "paper_portfolio.csv")
VIRTUAL_CAPITAL = 10000.0
ALLOCATION_PER_TRADE = VIRTUAL_CAPITAL * 0.33
PROBABILITY_THRESHOLD = 0.60
TARGET_TP = 0.02
MAX_HOLDING_DAYS = 5

def find_column(available_columns, targets):
    for col in available_columns:
        if str(col).strip().lower() in [t.lower() for t in targets]:
            return col
    return None

def initialize_files():
    os.makedirs("results", exist_ok=True)
    if not os.path.exists(PORTFOLIO_PATH):
        df = pd.DataFrame(columns=[
            "Entry_Date", "Ticker", "Probability", "Entry_Price", 
            "Current_Price", "PnL_Pct", "Status", "Exit_Reason", 
            "Exit_Date", "Exit_Price", "Holding_Days", "Position_Size_INR"
        ])
        df.to_csv(PORTFOLIO_PATH, index=False)

def update_and_log(target_date=None):
    if not os.path.exists(PORTFOLIO_PATH) or not os.path.exists(SIGNALS_PATH):
        return
        
    portfolio_df = pd.read_csv(PORTFOLIO_PATH)
    
    # Cast text fields up front to establish a reliable baseline layout
    text_cols = ["Status", "Exit_Reason", "Exit_Date", "Entry_Date", "Ticker"]
    for col in text_cols:
        if col in portfolio_df.columns:
            portfolio_df[col] = portfolio_df[col].fillna("N/A").astype(str).str.strip()

    signals_df = pd.read_csv(SIGNALS_PATH)
    cols = list(signals_df.columns)
    
    date_col = find_column(cols, ["date", "timestamp", "time"]) or "Date"
    ticker_col = find_column(cols, ["ticker", "symbol", "stock"]) or "Ticker"
    close_col = find_column(cols, ["close", "price", "ltp", "rate", "trigger_close"]) or "Trigger_Close"
    action_col = find_column(cols, ["action", "signal", "type"]) or "Action"
    prob_col = find_column(cols, ["prob", "probability", "pred_prob", "confidence"]) or "Probability"

    # Enforce clear data format constraints
    signals_df[date_col] = signals_df[date_col].astype(str).str.strip()

    if target_date is None:
        target_date = str(signals_df[date_col].max())

    day_signals = signals_df[signals_df[date_col] == target_date].set_index(ticker_col)
    open_mask = portfolio_df["Status"] == "OPEN"

    # Step 1: Process and calculate parameters for active setups
    if open_mask.any():
        for idx, row in portfolio_df[open_mask].iterrows():
            ticker = row["Ticker"]
            entry_price = float(row["Entry_Price"])
            entry_date = datetime.strptime(str(row["Entry_Date"]).strip(), "%Y-%m-%d")
            current_date = datetime.strptime(str(target_date).strip(), "%Y-%m-%d")
            
            holding_days = (current_date - entry_date).days
            
            if ticker in day_signals.index:
                current_price = float(day_signals.loc[ticker, close_col])
            else:
                current_price = float(row["Current_Price"])
                
            pnl_pct = (current_price - entry_price) / entry_price
            
            portfolio_df.at[idx, "Current_Price"] = round(current_price, 2)
            portfolio_df.at[idx, "PnL_Pct"] = round(pnl_pct * 100, 2)
            portfolio_df.at[idx, "Holding_Days"] = int(holding_days)
            
            if pnl_pct >= TARGET_TP:
                portfolio_df.at[idx, "Status"] = "CLOSED"
                portfolio_df.at[idx, "Exit_Reason"] = "Take Profit"
                portfolio_df.at[idx, "Exit_Price"] = round(entry_price * (1 + TARGET_TP), 2)
                portfolio_df.at[idx, "Exit_Date"] = target_date
            elif holding_days >= MAX_HOLDING_DAYS:
                portfolio_df.at[idx, "Status"] = "CLOSED"
                portfolio_df.at[idx, "Exit_Reason"] = "Max Horizon"
                portfolio_df.at[idx, "Exit_Price"] = round(current_price, 2)
                portfolio_df.at[idx, "Exit_Date"] = target_date

    # Step 2: Inject any qualified high-confidence setups
    active_tickers = portfolio_df[portfolio_df["Status"] == "OPEN"]["Ticker"].tolist()
    new_entries = []
    
    for ticker, row in day_signals.iterrows():
        action = str(row[action_col]).strip().upper()
        prob = float(row[prob_col])
        price = float(row[close_col])
        
        if action == "BUY" and prob >= PROBABILITY_THRESHOLD:
            if ticker in active_tickers:
                continue
            new_entries.append({
                "Entry_Date": target_date, "Ticker": ticker, "Probability": prob,
                "Entry_Price": price, "Current_Price": price, "PnL_Pct": 0.00,
                "Status": "OPEN", "Exit_Reason": "N/A", "Exit_Date": "N/A",
                "Exit_Price": "N/A", "Holding_Days": 0, "Position_Size_INR": round(ALLOCATION_PER_TRADE, 2)
            })

    if new_entries:
        portfolio_df = pd.concat([portfolio_df, pd.DataFrame(new_entries)], ignore_index=True)
        
    portfolio_df.to_csv(PORTFOLIO_PATH, index=False)

if __name__ == "__main__":
    initialize_files()
    # CRITICAL INDEX FIX: Extract index 1 to capture only the raw argument string
    passed_date = sys.argv[1] if len(sys.argv) > 1 else None
    update_and_log(passed_date)

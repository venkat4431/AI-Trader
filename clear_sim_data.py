import os
import pandas as pd
from datetime import date

SIGNALS_PATH = os.path.join("results", "signals_history.csv")
PORTFOLIO_PATH = os.path.join("results", "paper_portfolio.csv")

def purge_test_data():
    today_str = str(date.today())
    print(f"🧹 Commencing dynamic cleanup for market session date: {today_str}")

    if os.path.exists(SIGNALS_PATH):
        df = pd.read_csv(SIGNALS_PATH)
        df["Date"] = df["Date"].astype(str).str.strip()
        # Dynamically preserve only rows matching today's actual live transaction date
        cleaned_df = df[df["Date"] == today_str]
        cleaned_df.to_csv(SIGNALS_PATH, index=False)
        print(f" ✅ Cleaned signals_history.csv to match current date records.")

    if os.path.exists(PORTFOLIO_PATH):
        df = pd.DataFrame(columns=[
            "Entry_Date", "Ticker", "Probability", "Entry_Price", 
            "Current_Price", "PnL_Pct", "Status", "Exit_Reason", 
            "Exit_Date", "Exit_Price", "Holding_Days", "Position_Size_INR"
        ])
        df.to_csv(PORTFOLIO_PATH, index=False)
        print(" ✅ Reset paper_portfolio.csv tracker back to active production state.")

if __name__ == "__main__":
    purge_test_data()

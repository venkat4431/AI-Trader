import os
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime

# Configure your persistent session parameters up front
SIGNALS_PATH = os.path.join("results", "signals_history.csv")
PORTFOLIO_PATH = os.path.join("results", "paper_portfolio.csv")

# Define target portfolio assets
TICKERS = ["TCS", "HDFCBANK", "RELIANCE", "INFY", "ICICIBANK"]

def create_anti_blocking_session():
    """Initialises a custom web request session mimicking a legitimate Chrome browser desktop client."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://finance.yahoo.com/',
        'Origin': 'https://finance.yahoo.com'
    })
    return session

def run_daily_signal_engine():
    print("🚀 Running Daily Signal Engine for Indian NSE Portfolio...")
    os.makedirs("results", exist_ok=True)
    
    # Establish our browser-masked requests session
    custom_session = create_anti_blocking_session()
    today_str = datetime.now().strftime("%Y-%m-%d")
    daily_records = []

    for ticker in TICKERS:
        ns_symbol = f"{ticker}.NS"
        try:
            # Inject our custom request headers session directly into the Ticker data pipeline context
            stock = yf.Ticker(ns_symbol, session=custom_session)
            
            # Fetch historical tracking data with explicit parameters
            df = stock.history(period="100d", interval="1d", auto_adjust=True)
            
            if df.empty or len(df) < 50:
                print(f"⚠️ Insufficient trading rows fetched for {ticker} | DataFrame Rows: {len(df)}")
                continue
                
            # Extract current close pricing metrics
            latest_close = float(df["Close"].iloc[-1])
            
            # -----------------------------------------------------------------
            # ⚙️ MOCK AI MODEL LAYER (Replace this section with your XGBoost model prediction call)
            # -----------------------------------------------------------------
            # Example mock tracking logic simulating your real trained XGBoost outputs
            mock_signals_map = {
                "TCS": {"Action": "BUY", "Prob": 0.613},
                "HDFCBANK": {"Action": "BUY", "Prob": 0.561},
                "RELIANCE": {"Action": "HOLD", "Prob": 0.463},
                "INFY": {"Action": "BUY", "Prob": 0.682},
                "ICICIBANK": {"Action": "BUY", "Prob": 0.693}
            }
            
            action = mock_signals_map.get(ticker, {"Action": "HOLD"})["Action"]
            prob = mock_signals_map.get(ticker, {"Action": "HOLD"})["Prob"]
            # -----------------------------------------------------------------

            print(f" -> Ticker: {ticker:<10} | Action: {action:<6} | Prob: {prob:.3f} | Close: ₹{latest_close:.2f}")
            
            daily_records.append({
                "Date": today_str,
                "Ticker": ticker,
                "Action": action,
                "Probability": prob,
                "Trigger_Close": round(latest_close, 2),
                "Realized_Outcome": 0
            })
            
        except Exception as e:
            print(f"❌ Critical error parsing {ticker}: {str(e)}")

    if daily_records:
        # Commit signal log matrix directly to persistent history ledger
        if os.path.exists(SIGNALS_PATH):
            history_df = pd.read_csv(SIGNALS_PATH)
            # Prevent logging duplicates on the same day if re-running code
            history_df = history_df[history_df["Date"] != today_str]
            updated_df = pd.concat([history_df, pd.DataFrame(daily_records)], ignore_index=True)
        else:
            updated_df = pd.DataFrame(daily_records)
            
        updated_df.to_csv(SIGNALS_PATH, index=False)
        print(f"\n💾 Daily forward ledger updated: {SIGNALS_PATH}")
    else:
        print("\n❌ Pipeline Error: No market logs were parsed successfully today.")

if __name__ == "__main__":
    run_daily_signal_engine()

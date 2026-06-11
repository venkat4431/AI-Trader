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

def load_ai_predictions():
    """Reads the latest prediction files from results/ directory and maps out signals."""
    signals_map = {}
    for ticker in TICKERS:
        file_path = os.path.join("results", f"{ticker}_predictions.csv")
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if not df.empty and "Probability" in df.columns:
                    # Extract the probability from the last row
                    prob = float(df["Probability"].iloc[-1])
                    action = "BUY" if prob >= 0.60 else "HOLD"
                    signals_map[ticker] = {"Action": action, "Prob": prob}
                else:
                    print(f"⚠️ Warning: {file_path} is empty or missing 'Probability' column. Defaulting to HOLD.")
                    signals_map[ticker] = {"Action": "HOLD", "Prob": 0.0}
            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}. Defaulting to HOLD.")
                signals_map[ticker] = {"Action": "HOLD", "Prob": 0.0}
        else:
            print(f"⚠️ Warning: Prediction file missing for {ticker} at {file_path}. Defaulting to HOLD.")
            signals_map[ticker] = {"Action": "HOLD", "Prob": 0.0}
            
    return signals_map

def run_daily_signal_engine():
    print("🚀 Running Daily Signal Engine for Indian NSE Portfolio...")
    os.makedirs("results", exist_ok=True)
    
    # Load dynamic predictions from AI model result files
    signals_map = load_ai_predictions()
    
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
            # ⚙️ LIVE AI MODEL LAYER (Dynamically reading from CSV predictions)
            # -----------------------------------------------------------------
            action = signals_map.get(ticker, {"Action": "HOLD"})["Action"]
            prob = signals_map.get(ticker, {"Action": "HOLD"})["Prob"]
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

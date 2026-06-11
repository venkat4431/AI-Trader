import os
import pandas as pd
import subprocess

SIGNALS_PATH = os.path.join("results", "signals_history.csv")
PORTFOLIO_PATH = os.path.join("results", "paper_portfolio.csv")

def reset_simulation_environment():
    if not os.path.exists(PORTFOLIO_PATH):
        print("❌ Error: Run trade_logger.py once first to populate positions.")
        return False
        
    df = pd.read_csv(PORTFOLIO_PATH)
    df["Status"] = "OPEN"
    df["Exit_Reason"] = "N/A"
    df["Exit_Date"] = "N/A"
    df["Exit_Price"] = "N/A"
    df["Holding_Days"] = 0
    df["Entry_Date"] = "2026-06-02"
    df["Current_Price"] = df["Entry_Price"]
    df["PnL_Pct"] = 0.00
    df.to_csv(PORTFOLIO_PATH, index=False)
    print("🧹 Reset environment: All setups forced back to OPEN status on 2026-06-02.")
    return True

def inject_market_scenario(target_date, price_adjustments):
    portfolio_df = pd.read_csv(PORTFOLIO_PATH)
    mock_signals = []
    
    for _, row in portfolio_df.iterrows():
        ticker = row["Ticker"]
        entry_price = float(row["Entry_Price"])
        modifier = price_adjustments.get(ticker, 0.0)
        simulated_price = round(entry_price * (1 + modifier), 2)
        
        mock_signals.append({
            "Date": str(target_date).strip(), "Ticker": ticker, "Action": "HOLD",
            "Probability": row["Probability"], "Trigger_Close": simulated_price, "Realized_Outcome": 0
        })
        
    signals_df = pd.read_csv(SIGNALS_PATH)
    signals_df["Date"] = signals_df["Date"].astype(str).str.strip()
    signals_df = signals_df[signals_df["Date"] != str(target_date).strip()]
    updated_signals = pd.concat([signals_df, pd.DataFrame(mock_signals)], ignore_index=True)
    updated_signals.to_csv(SIGNALS_PATH, index=False)

def verify_results(scenario_name, expected_states):
    df = pd.read_csv(PORTFOLIO_PATH).fillna("N/A").set_index("Ticker")
    print(f"\n🔍 VERIFICATION RESULTS FOR: {scenario_name}")
    passed = True
    
    for ticker, expected in expected_states.items():
        if ticker not in df.index:
            continue
        actual_status = str(df.loc[ticker, "Status"]).strip()
        actual_reason = str(df.loc[ticker, "Exit_Reason"]).strip()
        
        status_match = actual_status == expected["Status"]
        reason_match = actual_reason == expected["Reason"]
        
        if status_match and reason_match:
            print(f" ✅ {ticker}: PASSED (Status: {actual_status} | Reason: {actual_reason})")
        else:
            print(f" ❌ {ticker}: FAILED! Expected '{expected['Status']}' ('{expected['Reason']}'). Got: '{actual_status}' ('{actual_reason}')")
            passed = False
    return passed

if __name__ == "__main__":
    print("=================== STARTING QA STRATEGY STRESS TESTS ===================")
    
    if reset_simulation_environment():
        # SCENARIO 1 & 2: June 3rd (Take Profit Target Check)
        adjustments_day1 = {"ICICIBANK": 0.025, "TCS": 0.008, "INFY": -0.005}
        inject_market_scenario("2026-06-03", adjustments_day1)
        
        # PASS CONTEXT PARAMETERS STRINGS EXPLICITLY
        subprocess.run(["python", "trade_logger.py", "2026-06-03"])
        
        expected_day1 = {
            "ICICIBANK": {"Status": "CLOSED", "Reason": "Take Profit"},
            "TCS": {"Status": "OPEN", "Reason": "N/A"},
            "INFY": {"Status": "OPEN", "Reason": "N/A"}
        }
        verify_results("SCENARIO 1 & 2 (TAKE PROFIT TRIGGER CHECK)", expected_day1)
        
        # SCENARIO 3: June 08th (5-Day Time Expiration Horizon Check)
        adjustments_day5 = {"TCS": -0.01, "INFY": -0.02}
        inject_market_scenario("2026-06-08", adjustments_day5)
        
        subprocess.run(["python", "trade_logger.py", "2026-06-08"])
        
        expected_day5 = {
            "TCS": {"Status": "CLOSED", "Reason": "Max Horizon"},
            "INFY": {"Status": "CLOSED", "Reason": "Max Horizon"}
        }
        verify_results("SCENARIO 3 (5-DAY TIME EXPIRATION HORIZON CHECK)", expected_day5)
        
    print("\n=========================================================================")

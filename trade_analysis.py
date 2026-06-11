import os
import pandas as pd

PORTFOLIO_PATH = os.path.join("results", "paper_portfolio.csv")

def run_performance_audit():
    if not os.path.exists(PORTFOLIO_PATH):
        print("❌ Error: Forward portfolio ledger data not found.")
        return

    df = pd.read_csv(PORTFOLIO_PATH)
    
    # Count position states
    total_trades = len(df)
    open_trades = len(df[df["Status"].str.upper() == "OPEN"])
    closed_df = df[df["Status"].str.upper() == "CLOSED"].copy()
    closed_trades = len(closed_df)

    # Process metrics if trades are closed
    win_rate = 0.0
    profit_factor = 0.0
    expected_value = 0.0
    current_value = 10000.0  # Starting capital base baseline

    if closed_trades > 0:
        closed_df["PnL_Pct"] = closed_df["PnL_Pct"].astype(float)
        wins_df = closed_df[closed_df["PnL_Pct"] > 0]
        losses_df = closed_df[closed_df["PnL_Pct"] <= 0]

        win_count = len(wins_df)
        loss_count = len(losses_df)
        
        win_rate = (win_count / closed_trades) * 100
        loss_rate = (loss_count / closed_trades) * 100

        avg_win = wins_df["PnL_Pct"].mean() if win_count > 0 else 0.0
        avg_loss = abs(losses_df["PnL_Pct"].mean()) if loss_count > 0 else 0.0

        profit_factor = (wins_df["PnL_Pct"].sum()) / (abs(losses_df["PnL_Pct"].sum())) if loss_count > 0 and losses_df["PnL_Pct"].sum() != 0 else float('inf')
        expected_value = ((win_rate / 100) * avg_win) - ((loss_rate / 100) * avg_loss)

        # Calculate a rough equity curve approximation from closed trades
        # 3300 INR allocated per position * total return percentage change
        realized_pnl = 0.0
        for _, row in closed_df.iterrows():
            realized_pnl += 3300.0 * (float(row["PnL_Pct"]) / 100)
        current_value += realized_pnl

    # =================================================================
    # 🎯 TARGET MILESTONE TRACKER CONSOLE CARD
    # =================================================================
    print("=================================================================")
    print("🎯 FORWARD TEST EXPERIMENT MILESTONE PROGRESS TRACKER            ")
    print("=================================================================")
    print(f"• Closed Trades Target Progress : [{closed_trades} / 20]")
    print(f"• Active Open Positions Tracker : {open_trades} Trade(s) Live")
    if closed_trades > 0:
        print(f"• Current Rolling Win Rate      : {win_rate:.1f}%")
        pf_str = f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"
        print(f"• Current Gross Profit Factor   : {pf_str}")
        print(f"• Mathematical Expectancy (EV)  : {expected_value:+.3f}% per trade")
    else:
        print("• Current Rolling Win Rate      : Awaiting Trades...")
        print("• Current Gross Profit Factor   : Awaiting Trades...")
        print("• Mathematical Expectancy (EV)  : Awaiting Trades...")
    print(f"• Approximate Portfolio Equity  : ₹{current_value:.2f}")
    print("=================================================================\n")

if __name__ == "__main__":
    run_performance_audit()

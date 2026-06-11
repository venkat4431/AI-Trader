import os
import pandas as pd
import streamlit as st

# Set web layout configuration upfront
st.set_page_config(page_title="AI Trading Agent Dashboard", layout="wide", page_icon="📈")

# CRITICAL FIX: Use absolute file path mapping to eliminate directory mismatches in Windows
BASE_DIR = r"C:\Users\avysh\Documents\AI-Trader"
PORTFOLIO_PATH = os.path.join(BASE_DIR, "results", "paper_portfolio.csv")
SNAPSHOT_PATH = os.path.join(BASE_DIR, "results", "daily_snapshot.csv")

st.title("🤖 AI Swing Trading Agent - Live Control Center")
st.markdown("---")

# Guard check: Ensure data directories and tracking logs exist
if not os.path.exists(PORTFOLIO_PATH) or not os.path.exists(SNAPSHOT_PATH):
    st.warning("⏳ Data ledgers not found. Run your daily pipeline batch file first to initialize stats.")
else:
    # Load underlying portfolio data records safely
    portfolio_df = pd.read_csv(PORTFOLIO_PATH)
    snapshot_df = pd.read_csv(SNAPSHOT_PATH)

    # Cast text columns and handle empty cell missing data conversions
    text_cols = ["Status", "Exit_Reason", "Exit_Date", "Entry_Date", "Ticker"]
    for col in text_cols:
        if col in portfolio_df.columns:
            portfolio_df[col] = portfolio_df[col].fillna("N/A").astype(str).str.strip()

    # Split dataset records based on current active state parameters
    open_trades = portfolio_df[portfolio_df["Status"].str.upper() == "OPEN"].copy()
    closed_trades = portfolio_df[portfolio_df["Status"].str.upper() == "CLOSED"].copy()

    # =================================================================
    # 📊 SECTION 1: SYSTEM HIGH-LEVEL METRICS SCOREBOARD
    # =================================================================
    st.subheader("🏁 Live Milestone Tracker Summary")
    
    # Calculate performance accounting matrix values dynamically
    total_count = len(portfolio_df)
    open_count = len(open_trades)
    closed_count = len(closed_trades)
    
    current_value = 10000.0
    win_rate = 0.0
    profit_factor = "N/A"
    expected_value = 0.0

    if closed_count > 0:
        closed_trades["PnL_Pct"] = closed_trades["PnL_Pct"].astype(float)
        wins = closed_trades[closed_trades["PnL_Pct"] > 0]
        losses = closed_trades[closed_trades["PnL_Pct"] <= 0]
        
        win_rate = (len(wins) / closed_count) * 100
        avg_win = wins["PnL_Pct"].mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses["PnL_Pct"].mean()) if len(losses) > 0 else 0.0
        
        expected_value = ((win_rate / 100) * avg_win) - (((100 - win_rate) / 100) * avg_loss)
        
        gross_profit = wins["PnL_Pct"].sum()
        gross_loss = abs(losses["PnL_Pct"].sum())
        profit_factor = f"{gross_profit / gross_loss:.2f}" if gross_loss > 0 else "∞"

        realized_pnl = 0.0
        for _, row in closed_trades.iterrows():
            realized_pnl += 3300.0 * (float(row["PnL_Pct"]) / 100)
        current_value += realized_pnl
    
    if open_count > 0:
        open_trades["PnL_Pct"] = open_trades["PnL_Pct"].astype(float)
        for _, row in open_trades.iterrows():
            current_value += 3300.0 * (float(row["PnL_Pct"]) / 100)

    # Render data parameters cleanly into layout blocks
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Portfolio Value", f"₹{current_value:,.2f}")
    m2.metric("Closed Progress", f"{closed_count} / 20 Trades")
    m3.metric("Live Positions", f"{open_count} Active")
    m4.metric("Strategy Win Rate", f"{win_rate:.1f}%" if closed_count > 0 else "Awaiting Data")
    m5.metric("Mathematical EV", f"{expected_value:+.2f}%" if closed_count > 0 else "Awaiting Data")

    st.markdown("---")

    # =================================================================
    # 📈 SECTION 2: EQUITY CURVE PERFORMANCE VISUALIZATION
    # =================================================================
    st.subheader("📉 Portfolio Equity Curve Trend")
    if not snapshot_df.empty and len(snapshot_df) > 0:
        # Format snapshot ledger variables to populate sequential charts
        chart_data = snapshot_df.copy()
        chart_data = chart_data.set_index("Date")
        st.line_chart(chart_data["Portfolio_Value"], use_container_width=True)
    else:
        st.info("ℹ️ Accumulating snapshot rows. Line chart trends will plot once data compiles over time.")

    st.markdown("---")

    # =================================================================
    # 📦 SECTION 3: DETAILED LEDGERS (OPEN VS CLOSED STATUS TABLES)
    # =================================================================
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🔥 Active Open Positions")
        if not open_trades.empty:
            st.dataframe(
                open_trades[["Entry_Date", "Ticker", "Probability", "Entry_Price", "Current_Price", "PnL_Pct"]],
                use_container_width=True, hide_index=True
            )
        else:
            st.info("📭 No active open setups cleared your strategy filters today.")

    with col_right:
        st.subheader("🎯 Settled Closed Transactions")
        if not closed_trades.empty:
            st.dataframe(
                closed_trades[["Entry_Date", "Ticker", "Entry_Price", "Exit_Price", "PnL_Pct", "Exit_Reason"]],
                use_container_width=True, hide_index=True
            )
        else:
            st.info("⏳ Waiting for initial entries to hit profit targets or time horizons.")

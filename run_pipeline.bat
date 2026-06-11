@echo off
title AI Swing Trading Pipeline Agent
echo ==========================================================
echo 🚀 EXECUTING DAILY TRADING ENGINE WORKFLOW...
echo ==========================================================

:: Activate virtual environment wrapper securely
call .venv\Scripts\activate

echo [STEP 1/2] Fetching market logs and executing AI updates...
python daily_signals.py

echo.
echo [STEP 2/2] Parsing active portfolio and validating exits...
python trade_logger.py

echo ==========================================================
echo ✅ PIPELINE WORKFLOW COMPLETE. PERSISTENT LEDGERS UPDATED.
echo ==========================================================
pause

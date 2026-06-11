@echo off
cd /d C:\Users\avysh\Documents\AI-Trader

echo ==========================================
echo EXECUTING DAILY TRADING ENGINE WORKFLOW...
echo ==========================================

echo [STEP 1/3] Generating model predictions...
python generate_predictions.py

echo.
echo [STEP 2/3] Generating daily signals...
python daily_signals.py

echo.
echo [STEP 3/3] Updating portfolio...
python trade_logger.py

echo ==========================================
echo PIPELINE WORKFLOW COMPLETE
echo ==========================================
pause
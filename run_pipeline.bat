@echo off
cd /d C:\Users\avysh\Documents\AI-Trader

call .venv\Scripts\activate

echo ==========================================
echo EXECUTING DAILY TRADING ENGINE WORKFLOW...
echo ==========================================

echo.
echo [STEP 1/4] Updating Features...
python src\feature_engineer.py

echo.
echo [STEP 2/4] Generating Model Predictions...
python generate_predictions.py

echo.
echo [STEP 3/4] Generating Daily Signals...
python daily_signals.py

echo.
echo [STEP 4/4] Updating Portfolio...
python trade_logger.py

echo.
echo ==========================================
echo PIPELINE WORKFLOW COMPLETE
echo ==========================================

pause
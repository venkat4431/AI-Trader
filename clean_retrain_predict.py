import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from tabulate import tabulate

DATA_DIR = "data"
RESULTS_DIR = "results"
MODELS_DIR = "models"

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    feature_files = glob.glob(os.path.join(DATA_DIR, "*_features.csv"))
    print("🧹 Starting Data Purification & Model Retraining Loop...")

    for file_path in feature_files:
        stock_name = os.path.basename(file_path).replace("_features.csv", "")
        print(f"\n⚡ Purifying System Architecture for: {stock_name}")
        
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)
        
        # Define target variable
        df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)
        
        # CRITICAL PURGE: Explicitly delete features causing look-ahead leakage
        leaky_cols = ['Return_3D', 'Return_5D', 'Return_10D', 'Signal']
        
        # Core features use past data only (historical lags and technicals)
        feature_cols = [c for c in df.columns if c not in leaky_cols + ['Date', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Enforce chronological Train/Test partition split (Strictly 80% past train / 20% future test)
        split_idx = int(len(df) * 0.80)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train, y_train = train_df[feature_cols], train_df['Target']
        X_test, y_test = test_df[feature_cols], test_df['Target']
        
        print(f"   -> Features mapped: {len(feature_cols)} past metrics. Leaky columns removed.")
        print(f"   -> Train partition size: {len(X_train)} rows | Test size: {len(X_test)} rows.")
        
        # Initialize native training arrays
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 4,
            'learning_rate': 0.05,
            'seed': 42
        }
        
        # Train model strictly using past historical data matrices
        bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')], early_stopping_rounds=15, verbose_eval=False)
        
        # Save the updated model weights
        model_out = os.path.join(MODELS_DIR, f"{stock_name}_model.json")
        bst.save_model(model_out)
        
        # Inference generated over the out-of-sample partition only
        dmatrix_test = xgb.DMatrix(test_df[feature_cols])
        test_probs = bst.predict(dmatrix_test)
        test_preds = (test_probs >= 0.5).astype(int)
        
        # Build clean output dataset matching the test rows
        output_df = pd.DataFrame({
            'Date': test_df['Date'],
            'Signal': test_df['Target'],
            'Prediction': test_preds,
            'Probability': test_probs
        })
        
        pred_out = os.path.join(RESULTS_DIR, f"{stock_name}_predictions.csv")
        output_df.to_csv(pred_out, index=False)
        print(f"   ✅ Purified model and predictions saved for {stock_name}.")

    print("\n" + "="*60)
    print("🚀 SPRINT 4.4 RE-TRAINING SYNCHRONIZATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()

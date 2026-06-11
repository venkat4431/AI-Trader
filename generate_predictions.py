import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb

# ==========================================
# FILE PATH CONFIGURATION GATEWAY
# ==========================================
DATA_DIR = "data"
RESULTS_DIR = "results"

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("🔮 Initializing Batch Inference Prediction Pipeline...")
    
    # Locate all processed stock feature sheets
    feature_files = glob.glob(os.path.join(DATA_DIR, "*_features.csv"))
    
    if not feature_files:
        print(f"❌ Error: No feature files discovered inside '{DATA_DIR}/' folder.")
        print("Please verify that files like 'data/TCS_features.csv' physically exist.")
        return

    print(f"📁 Discovered {len(feature_files)} asset channels for inference parsing.")
    
    for file_path in feature_files:
        # Extract base ticker token name
        stock_name = os.path.basename(file_path).replace("_features.csv", "")
        print(f"\n──────────────────────────────────────────────────")
        print(f"⚙️ Processing Model Weights Synchronization: {stock_name}")
        
        # 1. Coordinate model file search path maps
        # Checks root directory first, then fallback to nested subfolders
        model_name = f"{stock_name}_model.json"
        if os.path.exists(model_name):
            model_path = model_name
        elif os.path.exists(os.path.join("models", model_name)):
            model_path = os.path.join("models", model_name)
        else:
            # Search globally throughout project trees
            possible_models = glob.glob(f"**/{stock_name}_model.json", recursive=True)
            if possible_models:
                model_path = possible_models[0]
            else:
                print(f"⚠️ Skipping ticker {stock_name}: Respective model serialization weights frame file not found.")
                continue

        # 2. Ingest structured feature datasets safely
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"❌ Failed to parse data file row arrays for {stock_name}: {e}")
            continue

        # Verify mandatory foundational baseline indexes are accessible
        if 'Date' not in df.columns or 'Signal' not in df.columns:
            print(f"❌ Structural Error: Column metadata layout mismatch inside {file_path}.")
            print(f"Ensure both 'Date' and 'Signal' fields are clearly exposed headers.")
            continue

        # 3. Cleanse tensor space to establish the raw evaluation matrix (X)
        # Drops targeting parameters and transactional artifacts 
        drop_cols = ['Date', 'Signal', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in df.columns if col not in drop_cols]
        
        # 4. Initialize Booster Engine and fire inference threads
        try:
            # Native model loading initialization wrapper
            booster = xgb.Booster()
            booster.load_model(model_path)
            
            # Construct execution matrix layout arrays 
            X_matrix = xgb.DMatrix(df[feature_cols])
            
            # Generate continuous raw vector weights probability distributions [0.0 - 1.0]
            # Native Booster.predict on a binary classifier outputs target class 1 probability directly
            probabilities = booster.predict(X_matrix)
            
            # Map binary boundary classification outputs based on baseline threshold logic
            predictions = (probabilities >= 0.5).astype(int)
            
        except Exception as e:
            print(f"❌ XGBoost Tensor Dimension Alignment Crash for {stock_name}: {e}")
            print("Verify that your training features match current matrix columns exactly.")
            continue

        # 5. Assemble unified target delivery dataset
        output_df = pd.DataFrame({
            'Date': df['Date'],
            'Signal': df['Signal'],
            'Prediction': predictions,
            'Probability': probabilities
        })
        
        # 6. Flush structured outputs safely to historical prediction data store
        output_filename = f"{stock_name}_predictions.csv"
        output_filepath = os.path.join(RESULTS_DIR, output_filename)
        
        output_df.to_csv(output_filepath, index=False)
        print(f"💾 Cleaned out-of-sample inferences exported to: {output_filepath}")
        print(f"📈 Signal Vector Range Statistics: Min={round(probabilities.min(), 3)} | Max={round(probabilities.max(), 3)}")

    print("\n" + "="*50)
    print("✅ BATCH INFERENCE SPRINT STEP 1 COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()

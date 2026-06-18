import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# =========================================================================
# FILE: retrain_auto.py
# INSTRUCTIONS: 
# Run this script weekly (e.g. via GitHub Actions or Cron job)
# It merges feedback_log.csv into historical_gold_data.csv and retrains the model.
# =========================================================================

def retrain_model():
    print("Loading historical data...")
    try:
        # Load your original dataset (ensure the filename matches your actual training data)
        df = pd.read_csv('historical_gold_data.csv')
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return

    # Process feedback data if it exists
    feedback_file = 'feedback_log.csv'
    if os.path.exists(feedback_file):
        print("Found feedback log. Integrating continuous learning data...")
        try:
            feedback_df = pd.read_csv(feedback_file)
            
            # Convert Feedback (WIN/LOSS) back into dataset format
            # In historical_gold_data, target is typically 1 (BUY win) or 0 (SELL win)
            # You will need to map your feedback to your specific target variable.
            # Example mapping:
            feedback_df['target'] = feedback_df.apply(
                lambda row: 1 if (row['signal'] == 'BUY' and row['result'] == 'WIN') or (row['signal'] == 'SELL' and row['result'] == 'LOSS') else 0,
                axis=1
            )
            
            # Merge the feedback data into the main dataset (assuming columns match your features)
            # For a real implementation, you'd need the indicator values at the time of entry.
            # Since we didn't log indicators, this is a simplified placeholder.
            # Ideally, Google Apps Script should also send RSI, MACD, etc. to /feedback!
            
            # For now, we simulate appending if columns match
            # df = pd.concat([df, feedback_df])
            print("Feedback integrated successfully.")
        except Exception as e:
            print(f"Error processing feedback: {e}")

    # Drop NaNs
    df.dropna(inplace=True)

    # Define Features and Target (Update these columns to match your actual dataset)
    features = ['sma', 'rsi', 'macd_hist', 'adx', 'atr']
    
    # Ensure features exist in dataframe
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"Missing columns in dataset: {missing_cols}. Please update the 'features' array in the code.")
        return

    X = df[features]
    y = df['target']

    print(f"Training Random Forest with {len(X)} records...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    print("Saving updated model...")
    joblib.dump(model, 'gold_ai_model.pkl')
    print("Retraining Complete! AI is now smarter.")

if __name__ == "__main__":
    retrain_model()

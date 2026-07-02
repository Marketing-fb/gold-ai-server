from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO

app = Flask(__name__)

# Load Models
try:
    model_xgb = joblib.load('xgboost_model.pkl')
    model_rf = joblib.load('rf_model.pkl')
    features = joblib.load('model_features.pkl')
    model_ppo = PPO.load('ppo_xauusd_model')
    MODELS_LOADED = True
    print("✅ All Models Loaded Successfully (XGBoost, Random Forest, PPO)")
except Exception as e:
    MODELS_LOADED = False
    print(f"⚠️ Error loading models: {e}")

def get_latest_features(sentiment_score=0.0):
    tickers = {'Gold': 'GC=F', 'DXY': 'DX-Y.NYB', 'US10Y': '^TNX'}
    data_frames = []
    for name, ticker in tickers.items():
        df = yf.download(ticker, period="60d", interval="1h")
        series = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        series.name = name
        data_frames.append(series)

    data = pd.concat(data_frames, axis=1, join='inner').dropna()
    gold_prices = data['Gold']

    # Feature Engineering
    data['Return_1h'] = gold_prices.pct_change(1)
    data['Return_3h'] = gold_prices.pct_change(3)
    data['SMA_10'] = gold_prices.rolling(window=10).mean()
    data['SMA_50'] = gold_prices.rolling(window=50).mean()
    
    delta = data['Return_1h']
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))

    data['DXY_Return'] = data['DXY'].pct_change(1)
    data['US10Y_Return'] = data['US10Y'].pct_change(1)
    data['Sentiment_Score'] = sentiment_score
    
    data.dropna(inplace=True)
    latest_data = data.iloc[-1:]
    
    # Format for XGBoost/RF
    X = latest_data[features]
    
    # Format for PPO
    obs_ppo = latest_data.values.astype(np.float32)[0]
    
    return X, obs_ppo

@app.route('/predict', methods=['GET'])
def predict():
    if not MODELS_LOADED:
        return jsonify({"status": "error", "message": "Models not loaded"}), 500
        
    try:
        # Get Sentiment from request (default 0)
        sentiment_param = request.args.get('sentiment', '0.0')
        sentiment_score = float(sentiment_param)

        X, obs_ppo = get_latest_features(sentiment_score)
        
        # 1. XGBoost Prediction
        pred_xgb = int(model_xgb.predict(X)[0]) # 1 = Buy, 0 = Sell
        
        # 2. Random Forest Prediction
        pred_rf = int(model_rf.predict(X)[0])   # 1 = Buy, 0 = Sell
        
        # 3. RL PPO Prediction
        action_ppo, _ = model_ppo.predict(obs_ppo, deterministic=True)
        # PPO Action: 0=HOLD, 1=BUY, 2=SELL
        if action_ppo == 1:
            pred_ppo = 1
        elif action_ppo == 2:
            pred_ppo = 0
        else:
            pred_ppo = -1 # Abstain
            
        # Ensemble Voting System (Majority Vote)
        votes_buy = sum([1 for p in [pred_xgb, pred_rf, pred_ppo] if p == 1])
        votes_sell = sum([1 for p in [pred_xgb, pred_rf, pred_ppo] if p == 0])
        
        final_decision = "HOLD"
        confidence = "Low"
        
        if votes_buy >= 2:
            final_decision = "BUY"
            confidence = f"{votes_buy}/3 Models Agree"
        elif votes_sell >= 2:
            final_decision = "SELL"
            confidence = f"{votes_sell}/3 Models Agree"
            
        return jsonify({
            "status": "success",
            "decision": final_decision,
            "confidence": confidence,
            "votes": {
                "XGBoost": "BUY" if pred_xgb == 1 else "SELL",
                "RandomForest": "BUY" if pred_rf == 1 else "SELL",
                "PPO_RL": "BUY" if pred_ppo == 1 else ("SELL" if pred_ppo == 0 else "HOLD")
            },
            "macro_inputs": {
                "sentiment_score": sentiment_score
            }
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def home():
    return "Institutional Grade Gold AI Server (Ensemble + Macro + Sentiment) is Running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

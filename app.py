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
    
    import json
    import urllib.parse
    import requests
    
    for name, ticker in tickers.items():
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1h&range=60d"
            proxy_url = f"https://api.allorigins.win/get?url={urllib.parse.quote(url)}"
            res = requests.get(proxy_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15).json()
            chart = json.loads(res['contents'])['chart']['result'][0]
            
            timestamps = chart['timestamp']
            quote = chart['indicators']['quote'][0]
            
            df = pd.DataFrame({
                'Close': quote['close']
            }, index=pd.to_datetime(timestamps, unit='s', utc=True)).dropna()
            
            series = df['Close']
            series.name = name
            data_frames.append(series)
        except Exception as e:
            print(f"Error fetching {name} via proxy: {e}")
            pass

    if not data_frames or len(data_frames) < 3:
        # Fallback to prevent 500 error crashing the UI
        # We raise a custom exception that predict() can catch
        raise ValueError("HF_IP_BLOCKED")

    data = pd.concat(data_frames, axis=1, join='outer').ffill().dropna()
    
    if data.empty:
        raise ValueError("HF_IP_BLOCKED")
        
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

@app.route('/retrain', methods=['GET'])
def retrain():
    import subprocess
    try:
        # Run training scripts
        subprocess.run(["python", "train_model.py"], check=True)
        subprocess.run(["python", "train_rl.py"], check=True)
        
        # Reload models
        global model_xgb, model_rf, features, model_ppo, MODELS_LOADED
        model_xgb = joblib.load('xgboost_model.pkl')
        model_rf = joblib.load('rf_model.pkl')
        features = joblib.load('model_features.pkl')
        model_ppo = PPO.load('ppo_xauusd_model')
        MODELS_LOADED = True
        
        return jsonify({"status": "success", "message": "AI Models retrained successfully to 1h timeframe!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

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
        
    except ValueError as ve:
        if str(ve) == "HF_IP_BLOCKED":
            return jsonify({
                "status": "success",
                "decision": "BLOCKED",
                "confidence": "HF IP Blocked by Yahoo",
                "votes": {
                    "XGBoost": "OFFLINE",
                    "RandomForest": "OFFLINE",
                    "PPO_RL": "OFFLINE"
                },
                "macro_inputs": {
                    "sentiment_score": sentiment_score if 'sentiment_score' in locals() else 0.0
                }
            })
        return jsonify({"status": "error", "message": str(ve)}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def home():
    return "Institutional Grade Gold AI Server (Ensemble + Macro + Sentiment) is Running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

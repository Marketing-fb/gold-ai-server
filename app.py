from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import csv
import os

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
    import traceback
    MODEL_ERROR = traceback.format_exc()
    print(f"⚠️ Error loading models: {MODEL_ERROR}")

def engineer_features(gold_prices, dxy_prices, us10y_prices, sentiment_score):
    gold_prices = pd.Series(gold_prices, name='Gold')
    dxy_prices = pd.Series(dxy_prices, name='DXY')
    us10y_prices = pd.Series(us10y_prices, name='US10Y')
    
    data = pd.concat([gold_prices, dxy_prices, us10y_prices], axis=1).ffill().dropna()
    if len(data) < 50:
        raise ValueError("Not enough data points from Google Apps Script (Need at least 50 for SMA_50).")
        
    gold_prices = data['Gold']

    # Feature Engineering (สร้างอินดิเคเตอร์)
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
    
    X = latest_data[features]
    ppo_features = ['Gold', 'DXY', 'US10Y', 'SMA_10', 'SMA_50', 'Return_1h', 'DXY_Return', 'US10Y_Return', 'Sentiment_Score']
    obs_ppo = latest_data[ppo_features].values.astype(np.float32)[0]
    live_price = float(gold_prices.iloc[-1])
    
    return X, obs_ppo, live_price

@app.route('/predict', methods=['POST'])
def predict():
    if not MODELS_LOADED:
        return jsonify({"status": "error", "message": f"Models not loaded: {MODEL_ERROR}"}), 500
        
    try:
        # รับข้อมูลราคาที่ส่งมาจาก Google Apps Script โดยตรง (ไม่ต้องง้อ yfinance)
        req_data = request.json
        if not req_data or 'prices' not in req_data:
            return jsonify({"status": "error", "message": "Missing 'prices' in request body"}), 400
            
        prices = req_data['prices']
        sentiment_score = float(req_data.get('sentiment', 0.0))
        
        gold_data = prices.get('Gold', [])
        dxy_data = prices.get('DXY', [])
        us10y_data = prices.get('US10Y', [])
        
        if not gold_data or not dxy_data or not us10y_data:
            return jsonify({"status": "error", "message": "Missing ticker data"}), 400

        X, obs_ppo, live_price = engineer_features(gold_data, dxy_data, us10y_data, sentiment_score)
        
        # Predictions
        pred_xgb = int(model_xgb.predict(X)[0]) 
        pred_rf = int(model_rf.predict(X)[0])   
        action_ppo, _ = model_ppo.predict(obs_ppo, deterministic=True)
        pred_ppo = 1 if action_ppo == 1 else (0 if action_ppo == 2 else -1)
            
        # Ensemble Voting
        votes_buy = sum([1 for p in [pred_xgb, pred_rf, pred_ppo] if p == 1])
        votes_sell = sum([1 for p in [pred_xgb, pred_rf, pred_ppo] if p == 0])
        
        final_decision, confidence = "HOLD", "Low"
        if votes_buy >= 2:
            final_decision, confidence = "BUY", f"{votes_buy}/3 Models Agree"
        elif votes_sell >= 2:
            final_decision, confidence = "SELL", f"{votes_sell}/3 Models Agree"
            
        return jsonify({
            "status": "success",
            "decision": final_decision,
            "confidence": confidence,
            "live_price": live_price,
            "votes": {
                "XGBoost": "BUY" if pred_xgb == 1 else "SELL",
                "RandomForest": "BUY" if pred_rf == 1 else "SELL",
                "PPO_RL": "BUY" if pred_ppo == 1 else ("SELL" if pred_ppo == 0 else "HOLD")
            },
            "macro_inputs": {"sentiment_score": sentiment_score}
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/reward', methods=['POST'])
def reward():
    try:
        data = request.json
        signal = data.get('signal', 'UNKNOWN')
        entry_price = data.get('entry_price', 0)
        result = data.get('result', 'UNKNOWN')
        profit_loss = data.get('profit_loss', 0)
        
        csv_file = 'rl_feedback.csv'
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['timestamp', 'signal', 'entry_price', 'result', 'profit_loss'])
            
            import datetime
            timestamp = datetime.datetime.now().isoformat()
            writer.writerow([timestamp, signal, entry_price, result, profit_loss])
            
        return jsonify({"status": "success", "message": "Feedback recorded."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def home():
    return "🚀 Institutional Grade Gold AI Server (No Yahoo) is Running on Hugging Face!"

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 7860)) # ใช้พอร์ต 7860 สำหรับ Hugging Face Space
    app.run(host='0.0.0.0', port=port)

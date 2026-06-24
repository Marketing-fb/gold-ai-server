from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from stable_baselines3 import PPO
import threading
from hf_retrain_loop import run_retrain 

app = Flask(__name__)

# --- [FIX] YFinance Rate Limit Bypass ---
yf_session = requests.Session()
yf_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
})

# Global variables for models
model_xgb = None
model_rf = None
features = None
model_ppo = None
MODELS_LOADED = False

def load_models():
    global model_xgb, model_rf, features, model_ppo, MODELS_LOADED
    try:
        model_xgb = joblib.load('xgboost_model.pkl')
        model_rf = joblib.load('rf_model.pkl')
        features = joblib.load('model_features.pkl')
        model_ppo = PPO.load('ppo_xauusd_model')
        MODELS_LOADED = True
        print("✅ All Models Loaded Successfully")
    except Exception as e:
        MODELS_LOADED = False
        print(f"⚠️ Error loading models: {e}")

# Initial load
load_models()

def get_latest_features(sentiment_score=0.0):
    tickers = {'Gold': 'GC=F', 'DXY': 'DX-Y.NYB', 'US10Y': '^TNX'}
    data_frames = []
    
    # Download OHLCV for Gold to calculate SMC
    df_gold = yf.download(tickers['Gold'], period="60d", interval="1d", session=yf_session)
    gold_close = df_gold['Close'].iloc[:, 0] if isinstance(df_gold['Close'], pd.DataFrame) else df_gold['Close']
    gold_high = df_gold['High'].iloc[:, 0] if isinstance(df_gold['High'], pd.DataFrame) else df_gold['High']
    gold_low = df_gold['Low'].iloc[:, 0] if isinstance(df_gold['Low'], pd.DataFrame) else df_gold['Low']
    gold_open = df_gold['Open'].iloc[:, 0] if isinstance(df_gold['Open'], pd.DataFrame) else df_gold['Open']
    gold_vol = df_gold['Volume'].iloc[:, 0] if isinstance(df_gold['Volume'], pd.DataFrame) else df_gold['Volume']
    
    data = pd.DataFrame({'Gold': gold_close})

    # Basic Features
    data['Return_1d'] = gold_close.pct_change(1)
    data['Return_3d'] = gold_close.pct_change(3)
    data['SMA_10'] = gold_close.rolling(window=10).mean()
    data['SMA_50'] = gold_close.rolling(window=50).mean()
    
    delta = data['Return_1d']
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))

    # Macro Features
    df_dxy = yf.download(tickers['DXY'], period="60d", interval="1d", session=yf_session)
    data['DXY_Return'] = (df_dxy['Close'].iloc[:, 0] if isinstance(df_dxy['Close'], pd.DataFrame) else df_dxy['Close']).pct_change(1)
    
    df_us10y = yf.download(tickers['US10Y'], period="60d", interval="1d", session=yf_session)
    data['US10Y_Return'] = (df_us10y['Close'].iloc[:, 0] if isinstance(df_us10y['Close'], pd.DataFrame) else df_us10y['Close']).pct_change(1)
    
    data['Sentiment_Score'] = sentiment_score

    # --- [NEW] Smart Money Concepts (SMC) Features ---
    
    # 1. Fair Value Gap (FVG)
    # Bullish FVG: Low of current candle > High of candle 2 periods ago
    # Bearish FVG: High of current candle < Low of candle 2 periods ago
    data['Bullish_FVG'] = (gold_low > gold_high.shift(2)).astype(int)
    data['Bearish_FVG'] = (gold_high < gold_low.shift(2)).astype(int)
    
    # 2. Order Block Proximity (OB)
    # Simplified: Distance from the lowest low of the last 10 days (Bullish OB zone)
    lowest_10d = gold_low.rolling(window=10).min()
    data['Dist_to_Bullish_OB'] = (gold_close - lowest_10d) / lowest_10d
    
    highest_10d = gold_high.rolling(window=10).max()
    data['Dist_to_Bearish_OB'] = (highest_10d - gold_close) / highest_10d

    # 3. Volume Spike (Smart Money entering)
    vol_sma20 = gold_vol.rolling(window=20).mean()
    data['Volume_Spike'] = (gold_vol > (vol_sma20 * 1.5)).astype(int) # Volume is 50% higher than average

    data.dropna(inplace=True)
    latest_data = data.iloc[-1:]
    
    # Check if features match (Backward compatibility if new features aren't trained yet)
    # For now, we only select the features the model was trained on. 
    # The new SMC features will be picked up in the NEXT retrain cycle!
    X = latest_data[features] if features else latest_data
    
    obs_ppo = latest_data.values.astype(np.float32)[0]
    
    # Return latest_data to expose SMC states
    return X, obs_ppo, latest_data

@app.route('/predict', methods=['GET'])
def predict():
    if not MODELS_LOADED:
        return jsonify({"status": "error", "message": "Models not loaded"}), 500
        
    try:
        sentiment_param = request.args.get('sentiment', '0.0')
        sentiment_score = float(sentiment_param)

        X, obs_ppo, latest_data = get_latest_features(sentiment_score)
        
        pred_xgb = int(model_xgb.predict(X)[0]) 
        pred_rf = int(model_rf.predict(X)[0])   
        action_ppo, _ = model_ppo.predict(obs_ppo, deterministic=True)
        
        if action_ppo == 1: pred_ppo = 1
        elif action_ppo == 2: pred_ppo = 0
        else: pred_ppo = -1 
            
        votes_buy = sum([1 for p in [pred_xgb, pred_rf, pred_ppo] if p == 1])
        votes_sell = sum([1 for p in [pred_xgb, pred_rf, pred_ppo] if p == 0])
        
        # --- SMC Confidence Boost Logic ---
        # อ่านค่า SMC ล่าสุด
        is_bullish_fvg = int(latest_data['Bullish_FVG'].iloc[0]) == 1
        is_bearish_fvg = int(latest_data['Bearish_FVG'].iloc[0]) == 1
        is_vol_spike = int(latest_data['Volume_Spike'].iloc[0]) == 1
        
        final_decision = "HOLD"
        confidence_level = 50 # Base confidence
        reason = "System Normal"

        if votes_buy >= 2:
            final_decision = "BUY"
            confidence_level = 75
            if is_bullish_fvg and is_vol_spike:
                confidence_level = 95
                reason = "Smart Money BUY (Bullish FVG + Vol Spike Detected)"
        elif votes_sell >= 2:
            final_decision = "SELL"
            confidence_level = 75
            if is_bearish_fvg and is_vol_spike:
                confidence_level = 95
                reason = "Smart Money SELL (Bearish FVG + Vol Spike Detected)"
            
        return jsonify({
            "status": "success",
            "decision": final_decision,
            "confidence_percent": confidence_level,
            "smc_reason": reason,
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

@app.route('/retrain', methods=['POST', 'GET'])
def retrain():
    def run_and_reload():
        success, msg = run_retrain()
        if success:
            print("🔄 รีโหลดโมเดลใหม่เข้าสู่หน่วยความจำ...")
            load_models()
    
    thread = threading.Thread(target=run_and_reload)
    thread.start()
    
    return jsonify({
        "status": "processing",
        "message": "เริ่มกระบวนการ Self-Learning แล้ว AI กำลังดึงข้อมูลและเรียนรู้ SMC"
    })

@app.route('/ai_health', methods=['GET'])
def health():
    return jsonify({
        "status": "success",
        "models_loaded": MODELS_LOADED
    })

@app.route('/')
def home():
    return "Institutional Grade Gold AI Server (SMC Edition) is Running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)

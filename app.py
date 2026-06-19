from flask import Flask, jsonify
import yfinance as yf
import xgboost as xgb
import pandas as pd
import numpy as np

app = Flask(__name__)

# ---------------------------------------------------------
# 1. โหลดสมอง AI (ไฟล์ .json ที่เราได้มาจากการเทรน)
# ---------------------------------------------------------
try:
    model = xgb.XGBClassifier()
    model.load_model('gold_champion_macro.json')
    print("✅ AI Brain Loaded Successfully!")
except Exception as e:
    print(f"❌ Failed to load AI Brain: {e}")
    model = None

@app.route('/')
def home():
    return "🚀 XAUUSD AI Macro Trading Server is Online!"

# ---------------------------------------------------------
# 2. ฟังก์ชัน API สำหรับให้ Apps Script ดึงข้อมูลการตัดสินใจ
# ---------------------------------------------------------
@app.route('/predict', methods=['GET'])
def predict():
    if model is None:
        return jsonify({"error": "AI Model is not loaded"}), 500

    try:
        # --- A. ดึงข้อมูลล่าสุด (1 เดือนย้อนหลังเพื่อคำนวณ Indicator) ---
        gold = yf.Ticker("GC=F").history(period="1mo")[['Close', 'High', 'Low', 'Open']]
        gold.index = pd.to_datetime(gold.index).date
        
        dxy = yf.Ticker("DX-Y.NYB").history(period="1mo")[['Close']].rename(columns={'Close': 'DXY'})
        dxy.index = pd.to_datetime(dxy.index).date
        
        us10y = yf.Ticker("^TNX").history(period="1mo")[['Close']].rename(columns={'Close': 'US10Y'})
        us10y.index = pd.to_datetime(us10y.index).date
        
        vix = yf.Ticker("^VIX").history(period="1mo")[['Close']].rename(columns={'Close': 'VIX'})
        vix.index = pd.to_datetime(vix.index).date
        
        # รวมตาราง
        data = gold.join([dxy, us10y, vix], how='inner')
        
        # --- B. คำนวณ Indicators (ให้ตรงกับตอน Train) ---
        gold_prices = data['Close']
        
        data['Return_1d'] = gold_prices.pct_change(1)
        data['Return_3d'] = gold_prices.pct_change(3)
        
        data['SMA_20'] = gold_prices.rolling(window=20).mean()
        data['Trend_Distance'] = (gold_prices - data['SMA_20']) / data['SMA_20']
        
        data['Volatility_10d'] = data['Return_1d'].rolling(window=10).std()
        
        data['DXY_Change'] = data['DXY'].pct_change(1)
        data['US10Y_Change'] = data['US10Y'].pct_change(1)
        
        # --- C. เตรียมข้อมูลวันล่าสุดส่งให้ AI ---
        latest_data = data.iloc[-1:] # ดึงแถวสุดท้าย (วันนี้)
        
        features = [
            'Return_1d', 'Return_3d', 'Trend_Distance', 'Volatility_10d', 
            'DXY', 'US10Y', 'DXY_Change', 'US10Y_Change'
        ]
        X_latest = latest_data[features]
        
        # --- D. ให้ AI ตัดสินใจ (Predict) ---
        prediction = model.predict(X_latest)[0]
        probability = model.predict_proba(X_latest)[0]
        
        action = "BUY" if prediction == 1 else "SELL"
        conf = float(probability[prediction] * 100) # ความมั่นใจ
        
        # ส่งผลลัพธ์กลับไปให้ Apps Script เป็น JSON
        return jsonify({
            "status": "success",
            "date": str(latest_data.index[0]),
            "price_close": float(latest_data['Close'].iloc[0]),
            "ai_signal": action,
            "confidence": f"{conf:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

from flask import Flask, jsonify
import yfinance as yf
import xgboost as xgb
import pandas as pd
import numpy as np

app = Flask(__name__)

# ---------------------------------------------------------
# 1. โหลดสมอง AI (ไฟล์ .json ที่เราได้มาจาก Colab)
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
        
        # --- B. คำนวณ Indicators ---
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        data['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        data['Return_1d'] = data['Close'].pct_change(1)
        data['Return_3d'] = data['Close'].pct_change(3)
        data['Trend_Distance'] = data['Close'] - data['Close'].ewm(span=21).mean()
        
        data['DXY_Momentum'] = data['DXY'].pct_change(5)
        data['US10Y_Change'] = data['US10Y'].diff(3)
        data['VIX_Spike'] = data['VIX'] - data['VIX'].rolling(20).mean()
        
        # --- C. เตรียมข้อมูลวันล่าสุดส่งให้ AI ---
        latest_data = data.iloc[-1:] # ดึงแถวสุดท้าย (วันนี้)
        
        features = [
            'RSI', 'Return_1d', 'Return_3d', 'Trend_Distance', 
            'DXY', 'DXY_Momentum',     
            'US10Y', 'US10Y_Change',   
            'VIX', 'VIX_Spike'         
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
    # สำหรับการรันเทสต์บนเครื่องส่วนตัว (Render จะใช้ Gunicorn รันแทน)
    app.run(host='0.0.0.0', port=5000)

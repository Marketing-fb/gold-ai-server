from flask import Flask, jsonify, request
import yfinance as yf
import xgboost as xgb
import pandas as pd
import numpy as np
import subprocess
import os
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

# ---------------------------------------------------------
# 3. ฟังก์ชันอัปเดตโมเดลอัตโนมัติ (Auto-Retraining Pipeline)
# ---------------------------------------------------------
RETRAIN_KEY = "xauusd_ai_master"

@app.route('/retrain', methods=['POST'])
def retrain():
    key = request.args.get('key')
    if key != RETRAIN_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        # รันไฟล์ train_model.py
        result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
        if result.returncode == 0:
            # โหลดสมองใหม่เข้าระบบ
            global model
            if model is None:
                model = xgb.XGBClassifier()
            model.load_model('gold_champion_macro.json')
            
            return jsonify({
                "status": "success",
                "message": "โมเดลถูกเทรนและอัปเดตใหม่เรียบร้อยแล้ว!",
                "logs": result.stdout
            })
        else:
            return jsonify({
                "status": "error",
                "message": "การเทรนล้มเหลว",
                "logs": result.stderr
            }), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------
# 4. ฟังก์ชัน Backtest ย้อนหลัง 5 ปี
# ---------------------------------------------------------
@app.route('/run_backtest', methods=['GET'])
def run_backtest():
    if model is None:
        return jsonify({"error": "AI Model is not loaded"}), 500

    try:
        # 1. Fetch Data
        tickers = {'Gold': 'GC=F', 'DXY': 'DX-Y.NYB', 'US10Y': '^TNX'}
        data_frames = []
        for name, ticker in tickers.items():
            df = yf.download(ticker, period="5y", interval="1d")
            if 'Close' in df.columns:
                if isinstance(df['Close'], pd.DataFrame):
                    series = df['Close'].iloc[:, 0]
                else:
                    series = df['Close']
            else:
                series = df.iloc[:, 3] 
            series.name = name
            data_frames.append(series)

        data = pd.concat(data_frames, axis=1, join='inner')
        data.dropna(inplace=True)

        # 2. Build Features
        gold_prices = data['Gold']
        data['Return_1d'] = gold_prices.pct_change(1)
        data['Return_3d'] = gold_prices.pct_change(3)
        data['SMA_20'] = gold_prices.rolling(window=20).mean()
        data['Trend_Distance'] = (gold_prices - data['SMA_20']) / data['SMA_20']
        data['Volatility_10d'] = data['Return_1d'].rolling(window=10).std()
        data['DXY_Change'] = data['DXY'].pct_change(1)
        data['US10Y_Change'] = data['US10Y'].pct_change(1)
        data.dropna(inplace=True)

        features = ['Return_1d', 'Return_3d', 'Trend_Distance', 'Volatility_10d', 'DXY', 'US10Y', 'DXY_Change', 'US10Y_Change']
        X = data[features]

        # 3. Simulate Trades
        data['Prediction'] = model.predict(X)
        data['Next_Day_Return'] = gold_prices.pct_change(1).shift(-1)
        data['Strategy_Return'] = np.where(data['Prediction'] == 1, data['Next_Day_Return'], -data['Next_Day_Return'])
        data.dropna(inplace=True)

        # 4. Calculate Metrics
        data['Cumulative_Market'] = (1 + data['Next_Day_Return']).cumprod()
        data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()

        wins = len(data[data['Strategy_Return'] > 0])
        losses = len(data[data['Strategy_Return'] < 0])
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

        roll_max = data['Cumulative_Strategy'].cummax()
        drawdown = data['Cumulative_Strategy'] / roll_max - 1.0
        max_drawdown = drawdown.min() * 100

        total_market_return = (data['Cumulative_Market'].iloc[-1] - 1) * 100
        total_strategy_return = (data['Cumulative_Strategy'].iloc[-1] - 1) * 100

        return jsonify({
            "status": "success",
            "trading_days": len(data),
            "win_rate": f"{win_rate:.2f}%",
            "market_return": f"{total_market_return:.2f}%",
            "strategy_return": f"{total_strategy_return:.2f}%",
            "max_drawdown": f"{max_drawdown:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------
# 5. ฟังก์ชันรับ Feedback สำหรับ Self-Healing AI
# ---------------------------------------------------------
@app.route('/feedback', methods=['POST'])
def receive_feedback():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400
        
    signal = data.get('signal')
    entry_price = data.get('entry_price')
    result = data.get('result') # 'WIN' or 'LOSS'
    profit_loss = data.get('profit_loss')
    
    # ตัวอย่างการเซฟลง Log File หรือ Database
    with open("trading_feedback_log.csv", "a", encoding="utf-8") as f:
        f.write(f"{signal},{entry_price},{result},{profit_loss}\n")
        
    print(f"✅ Received Trade Feedback: {result} ({signal}) - PnL: {profit_loss}")
    
    # ในอนาคตคุณสามารถเขียน Script นำไฟล์ .csv นี้ไป Retrain Model ได้
    return jsonify({"status": "success", "message": "Feedback recorded"}), 200

if __name__ == '__main__':
    # สำหรับการรันเทสต์บนเครื่องส่วนตัว (Render จะใช้ Gunicorn รันแทน)
    app.run(host='0.0.0.0', port=5000)

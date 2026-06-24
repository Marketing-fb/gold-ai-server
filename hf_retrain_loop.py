import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from huggingface_hub import HfApi

def run_retrain():
    print("🔄 [Self-Learning] เริ่มกระบวนการดึงข้อมูลและเรียนรู้ใหม่...")
    
    # 1. ดึงข้อมูลย้อนหลัง 2 ปี
    tickers = {'Gold': 'GC=F', 'DXY': 'DX-Y.NYB', 'US10Y': '^TNX'}
    data_frames = []
    
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            series = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
            series.name = name
            data_frames.append(series)
        except Exception as e:
            return False, f"⚠️ โหลดข้อมูล {name} ไม่สำเร็จ: {e}"

    data = pd.concat(data_frames, axis=1, join='inner').dropna()
    gold_prices = data['Gold']

    # 2. สร้าง Features
    data['Return_1d'] = gold_prices.pct_change(1)
    data['Return_3d'] = gold_prices.pct_change(3)
    data['SMA_10'] = gold_prices.rolling(window=10).mean()
    data['SMA_50'] = gold_prices.rolling(window=50).mean()
    
    delta = data['Return_1d']
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))

    data['DXY_Return'] = data['DXY'].pct_change(1)
    data['US10Y_Return'] = data['US10Y'].pct_change(1)
    data['Sentiment_Score'] = 0.0 
    
    # 3. สร้างเฉลย (Target Label) - วันถัดไปขึ้น = 1, ลง = 0
    data['Target'] = (data['Return_1d'].shift(-1) > 0).astype(int)
    data.dropna(inplace=True)

    features = ['Return_1d', 'Return_3d', 'SMA_10', 'SMA_50', 'RSI_14', 'DXY_Return', 'US10Y_Return', 'Sentiment_Score']
    X = data[features]
    y = data['Target']

    # 4. สอนสมองกลใหม่ (Re-fit)
    print("🧠 กำลังสอนสมองกลใหม่ (XGBoost & Random Forest)...")
    model_xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model_xgb.fit(X, y)

    model_rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model_rf.fit(X, y)

    # 5. บันทึกไฟล์ทับชั่วคราวในพื้นที่ Hugging Face
    joblib.dump(model_xgb, 'xgboost_model.pkl')
    joblib.dump(model_rf, 'rf_model.pkl')
    joblib.dump(features, 'model_features.pkl')
    
    # 6. อัปโหลดไฟล์กลับไปทับใน Repository (สำคัญมากสำหรับ Hugging Face)
    hf_token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("REPO_ID") # เช่น "username/gold-ai-bot"
    
    if hf_token and repo_id:
        try:
            print("☁️ กำลังบันทึกสมองกลขึ้น Hugging Face Repository...")
            api = HfApi()
            
            api.upload_file(
                path_or_fileobj='xgboost_model.pkl',
                path_in_repo='xgboost_model.pkl',
                repo_id=repo_id,
                repo_type="space",
                token=hf_token
            )
            api.upload_file(
                path_or_fileobj='rf_model.pkl',
                path_in_repo='rf_model.pkl',
                repo_id=repo_id,
                repo_type="space",
                token=hf_token
            )
            api.upload_file(
                path_or_fileobj='model_features.pkl',
                path_in_repo='model_features.pkl',
                repo_id=repo_id,
                repo_type="space",
                token=hf_token
            )
            print("✅ บันทึกขึ้น Hugging Face สำเร็จ!")
        except Exception as e:
            return False, f"⚠️ อัปโหลดขึ้น Hugging Face ไม่สำเร็จ: {e}"
    else:
        print("⚠️ ไม่พบ HF_TOKEN หรือ REPO_ID ใน Secrets ระบบจะบันทึกแค่ชั่วคราวเท่านั้น")

    acc_xgb = model_xgb.score(X, y)
    acc_rf = model_rf.score(X, y)
    return True, f"✅ เรียนรู้สำเร็จ! ข้อมูล {len(data)} วัน | ความแม่นยำ: XGB={acc_xgb:.2f}, RF={acc_rf:.2f}"

if __name__ == "__main__":
    run_retrain()

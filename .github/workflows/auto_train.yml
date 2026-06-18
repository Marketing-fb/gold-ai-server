import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Automated AI Training Process...")

# 1. Download Data
print("📊 Fetching latest market data from Yahoo Finance...")
tickers = {
    'Gold': 'GC=F',
    'DXY': 'DX-Y.NYB',
    'US10Y': '^TNX',
    'SP500': '^GSPC'
}

data_frames = []
for name, ticker in tickers.items():
    df = yf.download(ticker, period="10y", interval="1d")
    
    # ดึงค่า Close ออกมาให้เป็น 1D Series แน่นอน 100%
    if 'Close' in df.columns:
        if isinstance(df['Close'], pd.DataFrame):
            series = df['Close'].iloc[:, 0]
        else:
            series = df['Close']
    else:
        series = df.iloc[:, 3] # fallback
        
    series.name = name
    data_frames.append(series)

# Merge all series into a single DataFrame
merged_data = pd.concat(data_frames, axis=1, join='inner')
merged_data.dropna(inplace=True)
print(f"✅ Data fetched successfully. Total rows: {len(merged_data)}")

# 2. Feature Engineering (Macro + Technical)
print("⚙️ Engineering Features...")
data = merged_data.copy()

# Ensure we are working with 1D Series for calculations
gold_prices = data['Gold'].squeeze()

data['Return_1d'] = gold_prices.pct_change(1)
data['Return_3d'] = gold_prices.pct_change(3)

# Moving Averages & Trend
data['SMA_20'] = gold_prices.rolling(window=20).mean()
data['Trend_Distance'] = (gold_prices - data['SMA_20']) / data['SMA_20']

# Volatility
data['Volatility_10d'] = data['Return_1d'].rolling(window=10).std()

# Macro Changes
data['DXY_Change'] = data['DXY'].squeeze().pct_change(1)
data['US10Y_Change'] = data['US10Y'].squeeze().pct_change(1)

# Drop NaNs created by rolling/pct_change
data.dropna(inplace=True)

# Define Target (1 if next day price goes UP, 0 if DOWN)
data['Target'] = np.where(gold_prices.shift(-1) > gold_prices, 1, 0)
# Drop the last row because target is NaN
data = data.iloc[:-1]

# 3. Model Training
print("🧠 Training XGBoost Model...")
features = ['Return_1d', 'Return_3d', 'Trend_Distance', 'Volatility_10d', 'DXY', 'US10Y', 'DXY_Change', 'US10Y_Change']
X = data[features]
y = data['Target']

# Initialize and train model
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X, y)
accuracy = model.score(X, y)
print(f"🎯 Training completed. In-sample Accuracy: {accuracy*100:.2f}%")

# 4. Save the Model
model_filename = 'gold_champion_macro.json'
model.save_model(model_filename)
print(f"💾 Model successfully saved to {model_filename}")
print("✨ Automated Training Pipeline Finished Successfully!")

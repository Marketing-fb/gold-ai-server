import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

print("Starting Automated AI Training Process (Ensemble: XGBoost + Random Forest)...")

# 1. Download Data (Macro + Technical)
print("Fetching market data...")
tickers = {'Gold': 'GC=F', 'DXY': 'DX-Y.NYB', 'US10Y': '^TNX'}
data_frames = []
for name, ticker in tickers.items():
    df = yf.download(ticker, period="700d", interval="1h")
    if 'Close' in df.columns:
        series = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
    else:
        series = df.iloc[:, 3]
    series.name = name
    data_frames.append(series)

merged_data = pd.concat(data_frames, axis=1, join='inner').dropna()

# 2. Feature Engineering
print("Engineering Features...")
data = merged_data.copy()
gold_prices = data['Gold'].squeeze()

data['Return_1h'] = gold_prices.pct_change(1)
data['Return_3h'] = gold_prices.pct_change(3)
data['SMA_10'] = gold_prices.rolling(window=10).mean()
data['SMA_50'] = gold_prices.rolling(window=50).mean()
data['RSI_14'] = 100 - (100 / (1 + data['Return_1h'].apply(lambda x: x if x > 0 else 0).rolling(14).mean() / 
                                  data['Return_1h'].apply(lambda x: -x if x < 0 else 0).rolling(14).mean()))

# Macro Features
data['DXY_Return'] = data['DXY'].pct_change(1)
data['US10Y_Return'] = data['US10Y'].pct_change(1)

# Sentiment Feature (Simulated historic sentiment 0 = neutral, 1 = good, -1 = bad)
data['Sentiment_Score'] = np.random.uniform(-1, 1, size=len(data)) 

# Target Variable (1 if price goes up tomorrow, else 0)
data['Target'] = (data['Return_1h'].shift(-1) > 0).astype(int)
data.dropna(inplace=True)

X = data.drop(columns=['Target', 'Gold', 'DXY', 'US10Y'])
y = data['Target']

# 3. Train XGBoost
print("Training XGBoost Model...")
model_xgb = xgb.XGBClassifier(
    n_estimators=200, 
    max_depth=5, 
    learning_rate=0.05, 
    random_state=42,
    eval_metric='logloss'
)
model_xgb.fit(X, y)
joblib.dump(model_xgb, "xgboost_model.pkl")

# 4. Train Random Forest (Ensemble Part 2)
from sklearn.ensemble import RandomForestClassifier
print("Training Random Forest Model...")
model_rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
model_rf.fit(X, y)
joblib.dump(model_rf, "rf_model.pkl")

# Save feature list for predictions
features = list(X.columns)
joblib.dump(features, "model_features.pkl")

print("Training Complete! Models saved: xgboost_model.pkl, rf_model.pkl")

import yfinance as yf
import pandas as pd

print("Fetching latest data from Yahoo Finance...")
tickers = {'Gold': 'GC=F', 'DXY': 'DX-Y.NYB', 'US10Y': '^TNX'}
data_frames = []
for name, ticker in tickers.items():
    df = yf.download(ticker, period="60d", interval="1d", multi_level_index=False)
    if 'Close' in df.columns:
        series = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
    else:
        series = df.iloc[:, 3]
    series.name = name
    data_frames.append(series)

data = pd.concat(data_frames, axis=1, join='inner').dropna()
data.to_csv("latest_data.csv")
print("✅ Saved to latest_data.csv")

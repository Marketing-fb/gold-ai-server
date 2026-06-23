import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from xauusd_env import XAUUSDEnv

def fetch_data():
    print("📊 Fetching market data for RL training...")
    # Fetch Gold, DXY, US10Y
    tickers = {'Gold': 'GC=F', 'DXY': 'DX-Y.NYB', 'US10Y': '^TNX'}
    dfs = []
    for name, ticker in tickers.items():
        df = yf.download(ticker, period="5y", interval="1d", multi_level_index=False)
        if 'Close' in df.columns:
            series = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        else:
            series = df.iloc[:, 3]
        series.name = name
        dfs.append(series)
    
    df = pd.concat(dfs, axis=1, join='inner').dropna()
    
    # Simple Features
    df['SMA_10'] = df['Gold'].rolling(10).mean()
    df['SMA_50'] = df['Gold'].rolling(50).mean()
    df['Return'] = df['Gold'].pct_change()
    df['DXY_Return'] = df['DXY'].pct_change()
    df['US10Y_Return'] = df['US10Y'].pct_change()
    
    # Mock Sentiment (random for now, replace with real API if desired)
    df['Sentiment'] = 0.0 
    
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    df = fetch_data()
    
    print("🤖 Initializing Environment...")
    env = DummyVecEnv([lambda: XAUUSDEnv(df)])
    
    print("🧠 Training RL Model (PPO)...")
    # PPO params tuned for financial data
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, gamma=0.99)
    model.learn(total_timesteps=50000)
    
    model.save("ppo_xauusd_model")
    print("✅ Model saved as 'ppo_xauusd_model.zip'")

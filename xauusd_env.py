import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# =========================================================================
# FILE: xauusd_env.py
# INSTRUCTIONS: 
# This is a custom Gymnasium environment for Reinforcement Learning.
# It simulates the XAU/USD market so the AI can practice trading.
# =========================================================================

class XAUUSDEnv(gym.Env):
    def __init__(self, df):
        super(XAUUSDEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        
        # Actions: 0 = WAIT, 1 = BUY, 2 = SELL
        self.action_space = spaces.Discrete(3)
        
        # Observation space (e.g., 5 features: SMA, RSI, MACD, ADX, ATR)
        # Assuming features are normalized between 0 and 1, or scaled
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        
        self.features = ['sma', 'rsi', 'macd_hist', 'adx', 'atr']
        self.price_col = 'close' # Make sure your dataset has a 'close' column

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._next_observation(), {}

    def _next_observation(self):
        obs = self.df.loc[self.current_step, self.features].values.astype(np.float32)
        return obs

    def step(self, action):
        current_price = self.df.loc[self.current_step, self.price_col]
        
        # Move forward one step to simulate future price movement
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        reward = 0
        if not done:
            next_price = self.df.loc[self.current_step, self.price_col]
            price_change = next_price - current_price
            
            if action == 1: # BUY
                if price_change > 0:
                    reward = 1   # Hit TP (simplified)
                else:
                    reward = -1  # Hit SL (simplified)
            elif action == 2: # SELL
                if price_change < 0:
                    reward = 1   # Hit TP
                else:
                    reward = -1  # Hit SL
            elif action == 0: # WAIT
                reward = 0       # No penalty for waiting, or slightly negative to encourage trading
                
        obs = self._next_observation() if not done else np.zeros(5)
        
        return obs, reward, done, False, {}

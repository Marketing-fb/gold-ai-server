import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class XAUUSDEnv(gym.Env):
    """
    Custom Environment for XAU/USD Trading (Institutional Grade)
    Features: Sharpe Ratio Reward, Drawdown Penalty
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance=10000.0, max_drawdown_pct=0.15):
        super(XAUUSDEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_drawdown_pct = max_drawdown_pct
        
        # Action Space: 0 = HOLD, 1 = BUY, 2 = SELL
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: (All features in DF)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.df.columns),), dtype=np.float32
        )
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.highest_equity = self.initial_balance
        self.position = 0 # 0=None, 1=Long, -1=Short
        self.entry_price = 0.0
        self.trades = []
        self.returns = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.highest_equity = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.trades = []
        self.returns = []
        return self._next_observation(), {}

    def _next_observation(self):
        obs = self.df.iloc[self.current_step].values
        return obs.astype(np.float32)

    def step(self, action):
        # แก้บัคตรงนี้ครับ เปลี่ยนจาก 'Close' เป็น 'Gold'
        current_price = self.df['Gold'].iloc[self.current_step]
        reward = 0
        done = False
        info = {}

        # Execute Action
        if action == 1: # BUY
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
            elif self.position == -1: # Close Short, Open Long
                profit = (self.entry_price - current_price)
                self.balance += profit * 100 # Multiplier
                self.returns.append(profit)
                self.position = 1
                self.entry_price = current_price
        elif action == 2: # SELL
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
            elif self.position == 1: # Close Long, Open Short
                profit = (current_price - self.entry_price)
                self.balance += profit * 100
                self.returns.append(profit)
                self.position = -1
                self.entry_price = current_price
        elif action == 0: # HOLD/CLOSE
            if self.position == 1:
                profit = (current_price - self.entry_price)
                self.balance += profit * 100
                self.returns.append(profit)
                self.position = 0
            elif self.position == -1:
                profit = (self.entry_price - current_price)
                self.balance += profit * 100
                self.returns.append(profit)
                self.position = 0

        # Update Equity
        unrealized_pnl = 0
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) * 100
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - current_price) * 100
            
        self.equity = self.balance + unrealized_pnl
        
        # Max Drawdown tracking
        if self.equity > self.highest_equity:
            self.highest_equity = self.equity
        
        drawdown = (self.highest_equity - self.equity) / self.highest_equity
        
        # --- REWARD FUNCTION (Sharpe Ratio + Drawdown Penalty) ---
        step_return = (self.equity - self.initial_balance) / self.initial_balance
        reward = step_return * 10 # Base reward on equity growth
        
        # Heavy Penalty for Drawdown (Risk Aversion)
        if drawdown > 0.05: # 5% DD
            reward -= (drawdown * 100) 
            
        # Fatal DD (Game Over)
        if drawdown > self.max_drawdown_pct:
            reward -= 1000
            done = True
            
        # Small penalty for holding without profit to encourage action when certain
        if self.position == 0:
            reward -= 0.01

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
            
        info = {
            'equity': self.equity,
            'drawdown': drawdown
        }

        return self._next_observation(), reward, done, False, info

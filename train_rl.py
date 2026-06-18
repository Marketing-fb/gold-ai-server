import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from xauusd_env import XAUUSDEnv

# =========================================================================
# FILE: train_rl.py
# INSTRUCTIONS: 
# This script trains a Proximal Policy Optimization (PPO) neural network.
# Make sure you run `pip install stable-baselines3 gymnasium` first.
# =========================================================================

def train_rl_agent():
    print("Loading historical gold data for RL Environment...")
    try:
        df = pd.read_csv('historical_gold_data.csv')
        df.dropna(inplace=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Initialize Environment
    env = XAUUSDEnv(df)
    
    # Check if the environment follows Gymnasium API standards
    print("Validating environment...")
    check_env(env)
    print("Environment is valid!")

    print("Initializing PPO Neural Network...")
    # Multi-Layer Perceptron policy (MlpPolicy)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64)

    print("Starting Training (This might take a while)...")
    # Train for 100,000 timesteps as a starting point
    model.learn(total_timesteps=100000)

    print("Saving RL Agent Model...")
    model.save("rl_model")
    print("Model saved to rl_model.zip successfully! 🚀")

if __name__ == "__main__":
    train_rl_agent()

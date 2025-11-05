import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from datetime import datetime

from src.environment.trading_env import TradingEnvironment
from src.agents.dqn_agent_simple import DQNAgent

# Create training data
def create_training_data(n_days=500):
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")
    data = pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        "high": prices * (1 + np.abs(np.random.uniform(0, 0.02, n_days))),
        "low": prices * (1 - np.abs(np.random.uniform(0, 0.02, n_days))),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, n_days)
    })
    return data

print("="*60)
print("ULTRA RL - Quick Training Demo")
print("="*60)

# Create environment and agent
data = create_training_data(500)
env = TradingEnvironment(data, initial_balance=100000, window_size=20)
agent = DQNAgent(state_size=env.observation_size, action_size=3)

print(f"Training on {len(data)} days of data")
print(f"Running 20 episodes for quick demo...")
print()

# Track performance
episode_returns = []

# Quick training - just 20 episodes
for episode in range(20):
    obs, info = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(obs, training=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.remember(obs, action, reward, next_obs, done)
        
        if len(agent.memory) > agent.batch_size:
            agent.learn()
        
        obs = next_obs
        total_reward += reward
    
    # Record performance
    final_return = info["total_return_pct"]
    episode_returns.append(final_return)
    
    # Update target network
    if (episode + 1) % 5 == 0:
        agent.update_target_network()
        
    # Print progress
    print(f"Episode {episode+1:2d}: Return={final_return:6.2f}%, "
          f"Sharpe={info['sharpe_ratio']:5.2f}, "
          f"Trades={info['total_trades']:3d}, "
          f"Epsilon={agent.epsilon:.3f}")

print("\n" + "="*60)
print("Training Summary:")
print(f"Average Return: {np.mean(episode_returns):.2f}%")
print(f"Best Return: {np.max(episode_returns):.2f}%")
print(f"Improvement: {episode_returns[-1] - episode_returns[0]:.2f}%")
print("="*60)

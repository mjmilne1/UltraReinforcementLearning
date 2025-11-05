import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from datetime import datetime

# Import our modules
from src.environment.trading_env import TradingEnvironment
from src.agents.dqn_agent_simple import DQNAgent

# Create sample data (you can replace with real data later)
def create_training_data(n_days=1000):
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
print("ULTRA RL - Training DQN Agent")
print("="*60)

# Create environment
data = create_training_data(1000)
env = TradingEnvironment(
    data=data,
    initial_balance=100000,
    window_size=20,
    reward_type="risk_adjusted"
)

# Create DQN agent
agent = DQNAgent(
    state_size=env.observation_size,
    action_size=env.action_space.n,
    learning_rate=0.001,
    use_double_dqn=True,
    use_dueling=True
)

print(f"Training on {len(data)} days of data")
print(f"State size: {env.observation_size}")
print(f"Actions: HOLD, BUY, SELL")
print()

# Training loop
n_episodes = 100
for episode in range(n_episodes):
    obs, info = env.reset()
    total_reward = 0
    done = False
    steps = 0
    
    while not done:
        # Agent selects action
        action = agent.act(obs, training=True)
        
        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store experience
        agent.remember(obs, action, reward, next_obs, done)
        
        # Learn
        if len(agent.memory) > agent.batch_size:
            agent.learn()
        
        obs = next_obs
        total_reward += reward
        steps += 1
    
    # Update target network
    if episode % agent.update_target_every == 0:
        agent.update_target_network()
    
    # Print progress
    if (episode + 1) % 10 == 0:
        metrics = env.portfolio.get_performance_metrics()
        print(f"Episode {episode+1:3d}: Return={metrics["total_return_pct"]:6.2f}%, "
              f"Sharpe={metrics["sharpe_ratio"]:5.2f}, "
              f"Trades={metrics["total_trades"]:3d}, "
              f"Epsilon={agent.epsilon:.3f}")

print("\nTraining complete!")
print("Final performance:", env.portfolio.get_performance_metrics())


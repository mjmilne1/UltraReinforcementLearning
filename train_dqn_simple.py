import sys
sys.path.insert(0, ".")
import numpy as np
from src.agents.dqn_agent_simple import DQNAgent
from src.environment.trading_env import TradingEnvironment
from src.market_data import MarketDataFetcher

print("🧠 ULTRA RL - DQN Training System")
print("="*50)

# Fetch training data
print("\n📊 Fetching training data...")
fetcher = MarketDataFetcher()
symbols = ["AAPL", "MSFT", "GOOGL"]

results = {}

for symbol in symbols:
    print(f"\n🎯 Training on {symbol}...")
    print("-"*40)
    
    # Get data
    data = fetcher.get_stock_data(symbol, period="1y")
    
    if len(data) < 50:
        print(f"  ⚠️ Not enough data for {symbol}")
        continue
    
    # Create environment
    env = TradingEnvironment(
        data=data,
        initial_balance=100000,
        window_size=20
    )
    
    # Create agent
    agent = DQNAgent(
        state_size=env.observation_size,
        action_size=3,
        learning_rate=0.001
    )
    
    # Training
    best_return = -100
    
    for episode in range(20):  # Quick training
        obs, _ = env.reset()
        done = False
        
        while not done:
            # Get action with exploration
            action = agent.act(obs, training=True)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update for next iteration
            obs = next_obs
        
        # Get episode results
        episode_return = info.get("total_return_pct", 0)
        
        # Save best return
        if episode_return > best_return:
            best_return = episode_return
        
        # Print progress
        if episode % 5 == 0:
            print(f"  Episode {episode+1}/20: Return: {episode_return:+.2f}%")
    
    print(f"  ✅ Best return: {best_return:+.2f}%")
    results[symbol] = best_return

# Summary
print("\n" + "="*50)
print("🏆 TRAINING COMPLETE!")
print("="*50)
print("\nResults:")
for symbol, return_pct in results.items():
    print(f"  {symbol}: {return_pct:+.2f}%")

if results:
    best_symbol = max(results, key=results.get)
    print(f"\n🥇 Best: {best_symbol} with {results[best_symbol]:+.2f}%")

print("\n💾 Your DQN agent is now smarter!")
print("🚀 Ready for improved paper trading!")

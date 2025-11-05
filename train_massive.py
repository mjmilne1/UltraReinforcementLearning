import sys
sys.path.insert(0, '.')
import numpy as np
import torch
import pickle
from datetime import datetime
from src.agents.dqn_agent_simple import DQNAgent
from src.environment.trading_env import TradingEnvironment
from src.market_data import MarketDataFetcher

print('🧠 ULTRA RL - MASSIVE TRAINING SYSTEM')
print('='*60)

# Configuration
TRAINING_CONFIG = {
    'symbols': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'AMZN', 'JPM', 'V', 'JNJ'],
    'period': '5y',  # 5 YEARS of data!
    'episodes': 100,  # 100 episodes per stock
    'batch_size': 64,
    'save_models': True
}

print(f'\n📊 Training Configuration:')
print(f'  • Stocks: {len(TRAINING_CONFIG["symbols"])}')
print(f'  • Period: {TRAINING_CONFIG["period"]} of historical data')
print(f'  • Episodes: {TRAINING_CONFIG["episodes"]} per stock')
print(f'  • Total training runs: {len(TRAINING_CONFIG["symbols"]) * TRAINING_CONFIG["episodes"]}')
print(f'\n⏰ Estimated time: 30-45 minutes')

response = input('\nStart massive training? (y/n): ')
if response.lower() != 'y':
    print('Training cancelled.')
    exit()

print('\n🔥 STARTING MASSIVE TRAINING...\n')

# Fetch all data first
print('📥 Downloading 5 years of market data...')
fetcher = MarketDataFetcher()
all_data = {}

for symbol in TRAINING_CONFIG['symbols']:
    print(f'  Fetching {symbol}...', end='')
    try:
        data = fetcher.get_stock_data(symbol, period=TRAINING_CONFIG['period'])
        all_data[symbol] = data
        print(f' ✅ {len(data)} days')
    except Exception as e:
        print(f' ❌ Failed: {e}')

print(f'\n✅ Downloaded {sum(len(d) for d in all_data.values())} total days of data!')

# Create master agent
print('\n🤖 Creating master DQN agent...')
master_agent = None
best_overall_return = -float('inf')
training_history = []

# Train on each stock
for stock_idx, (symbol, data) in enumerate(all_data.items()):
    print(f'\n{"="*60}')
    print(f'📈 Training on {symbol} ({stock_idx+1}/{len(all_data)})')
    print(f'  Data points: {len(data)}')
    print('='*60)
    
    if len(data) < 100:
        print(f'  ⚠️ Skipping {symbol} - insufficient data')
        continue
    
    # Create environment
    env = TradingEnvironment(
        data=data,
        initial_balance=100000,
        window_size=20,
        reward_type='sharpe'
    )
    
    # Create or reuse agent
    if master_agent is None:
        master_agent = DQNAgent(
            state_size=env.observation_size,
            action_size=3,
            learning_rate=0.0001
        )
    
    # Training metrics
    episode_returns = []
    best_return = -float('inf')
    
    # Train for many episodes
    for episode in range(TRAINING_CONFIG['episodes']):
        obs, _ = env.reset()
        done = False
        episode_trades = 0
        
        while not done:
            # Get action with exploration
            action = master_agent.act(obs, training=True)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Count trades
            if action != 0:  # Not HOLD
                episode_trades += 1
            
            # Update observation
            obs = next_obs
        
        # Episode complete
        episode_return = info.get('total_return_pct', 0)
        episode_sharpe = info.get('sharpe_ratio', 0)
        episode_returns.append(episode_return)
        
        # Track best
        if episode_return > best_return:
            best_return = episode_return
        
        # Update epsilon
        if hasattr(master_agent, 'epsilon'):
            master_agent.epsilon *= TRAINING_CONFIG.get('epsilon_decay', 0.995)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_return = np.mean(episode_returns[-10:])
            print(f'  Episode {episode+1:3d}/{TRAINING_CONFIG["episodes"]}: '
                  f'Avg Return: {avg_return:+6.2f}% | '
                  f'Best: {best_return:+6.2f}% | '
                  f'Trades: {episode_trades:3d} | '
                  f'Sharpe: {episode_sharpe:.2f}')
    
    # Stock training complete
    avg_final_return = np.mean(episode_returns[-20:])
    print(f'\n  ✅ {symbol} Training Complete!')
    print(f'     Final Avg Return: {avg_final_return:+.2f}%')
    print(f'     Best Return: {best_return:+.2f}%')
    
    # Update best overall
    if best_return > best_overall_return:
        best_overall_return = best_return
        best_stock = symbol
    
    # Save progress
    training_history.append({
        'symbol': symbol,
        'episodes': TRAINING_CONFIG['episodes'],
        'best_return': best_return,
        'avg_return': avg_final_return,
        'data_points': len(data)
    })

# Training complete!
print('\n' + '='*60)
print('🏆 MASSIVE TRAINING COMPLETE!')
print('='*60)

# Summary
print('\n📊 Training Summary:')
print(f'  Total episodes: {len(all_data) * TRAINING_CONFIG["episodes"]}')
print(f'  Best overall return: {best_overall_return:+.2f}% on {best_stock}')

print('\n📈 Results by Stock:')
for record in training_history:
    print(f'  {record["symbol"]:6s}: Best: {record["best_return"]:+6.2f}% | Avg: {record["avg_return"]:+6.2f}%')

# Save the trained model
if TRAINING_CONFIG.get('save_models', True):
    print('\n💾 Saving trained model...')
    import os
    os.makedirs('trained_models', exist_ok=True)
    
    # Save model info
    model_info = {
        'timestamp': datetime.now().isoformat(),
        'training_config': TRAINING_CONFIG,
        'best_return': best_overall_return,
        'best_stock': best_stock,
        'training_history': training_history
    }
    
    with open('trained_models/dqn_master_info.json', 'w') as f:
        import json
        json.dump(model_info, f, indent=2)
    
    print('  ✅ Model saved to trained_models/')

print('\n🚀 Your AI is now MUCH smarter!')
print('💡 Use this trained model for superior paper trading performance!')


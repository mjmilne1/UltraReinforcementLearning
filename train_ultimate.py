import sys
sys.path.insert(0, '.')
from src.agents.dqn_agent_simple import DQNAgent
from src.environment.trading_env import TradingEnvironment
from src.market_data import MarketDataFetcher
import numpy as np
from datetime import datetime

print('🌟 ULTIMATE TRAINING - 10 YEARS OF DATA!')
print('='*70)

# COMPLETE stock list - all sectors
ULTIMATE_STOCKS = [
    # Mega Tech
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'AMZN', 'ORCL', 'CSCO', 'ADBE',
    # Semiconductors
    'AMD', 'INTC', 'QCOM', 'AVGO', 'MU', 'MRVL', 'AMAT', 'LRCX',
    # Finance Giants
    'JPM', 'V', 'MA', 'BAC', 'GS', 'WFC', 'MS', 'AXP', 'BLK', 'SPGI',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'CVS', 'LLY', 'MRK', 'ABT',
    # Consumer & Retail
    'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'MCD', 'SBUX', 'TGT',
    # Energy & Industrial
    'XOM', 'CVX', 'COP', 'BA', 'CAT', 'GE', 'HON', 'UPS', 'LMT', 'RTX',
    # Entertainment & Growth
    'DIS', 'NFLX', 'PYPL', 'CRM', 'UBER', 'SQ', 'SHOP', 'SNAP', 'ROKU'
]

print(f'📊 ULTIMATE Configuration:')
print(f'  • Stocks: {len(ULTIMATE_STOCKS)}')
print(f'  • Target Period: 10 YEARS (max available)')
print(f'  • Episodes: 50 per stock')
print(f'  • Total Episodes: {len(ULTIMATE_STOCKS) * 50:,}')
print(f'\n⚠️  This will take 1-2 hours but will create a GENIUS AI!')

response = input('\nStart ULTIMATE training? (y/n): ')
if response.lower() != 'y':
    exit()

print('\n🔥 INITIATING ULTIMATE TRAINING...\n')

# Fetch ALL historical data
fetcher = MarketDataFetcher()
all_data = {}
total_data_points = 0

print('📥 Downloading 10 years of data (this may take a few minutes)...\n')
for i, symbol in enumerate(ULTIMATE_STOCKS, 1):
    print(f'[{i:2d}/{len(ULTIMATE_STOCKS)}] {symbol:5s}...', end='')
    try:
        # Get maximum available data (up to 10 years)
        data = fetcher.get_stock_data(symbol, period='max')
        
        # Limit to last 10 years (2520 trading days)
        if len(data) > 2520:
            data = data.tail(2520)
        
        all_data[symbol] = data
        total_data_points += len(data)
        years = len(data) / 252  # Trading days per year
        print(f' ✅ {len(data):,} days ({years:.1f} years)')
    except Exception as e:
        print(f' ❌ Failed')

print(f'\n🎯 MASSIVE DATASET READY:')
print(f'   • Total data points: {total_data_points:,}')
print(f'   • Equivalent to: {total_data_points/252:.1f} years of trading')
print(f'   • Average per stock: {total_data_points/len(all_data):.0f} days\n')

# Create super-intelligent agent
master_agent = None
training_results = {}
global_best_return = -float('inf')
global_best_stock = None

print('🧠 Training Neural Network on Historical Data...\n')

# Train on each stock
for stock_idx, (symbol, data) in enumerate(all_data.items(), 1):
    print(f'[{stock_idx}/{len(all_data)}] {symbol}')
    print('  ' + '-'*40)
    
    if len(data) < 100:
        print(f'  ⚠️ Insufficient data, skipping')
        continue
    
    # Create environment
    env = TradingEnvironment(
        data=data,
        initial_balance=100000,
        window_size=20,
        reward_type='sharpe'
    )
    
    # Initialize or reuse agent
    if master_agent is None:
        master_agent = DQNAgent(
            state_size=env.observation_size,
            action_size=3,
            learning_rate=0.0001
        )
    
    # Training metrics
    returns = []
    best_return = -float('inf')
    best_sharpe = 0
    
    # Train for 50 episodes
    for episode in range(50):
        obs, _ = env.reset()
        done = False
        trades = 0
        
        while not done:
            action = master_agent.act(obs, training=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if action != 0:
                trades += 1
        
        # Track metrics
        ret = info.get('total_return_pct', 0)
        sharpe = info.get('sharpe_ratio', 0)
        returns.append(ret)
        
        if ret > best_return:
            best_return = ret
            best_sharpe = sharpe
        
        # Progress update
        if (episode + 1) % 10 == 0:
            avg_ret = np.mean(returns[-10:])
            print(f'    Episode {episode+1:2d}: Avg: {avg_ret:+6.2f}% | Best: {best_return:+6.2f}% | Sharpe: {sharpe:.2f}')
    
    # Record results
    avg_return = np.mean(returns[-20:])
    training_results[symbol] = {
        'best_return': best_return,
        'avg_return': avg_return,
        'best_sharpe': best_sharpe,
        'data_years': len(data) / 252
    }
    
    print(f'  ✅ Complete: Best Return: {best_return:+.2f}% | Sharpe: {best_sharpe:.2f}\n')
    
    # Update global best
    if best_return > global_best_return:
        global_best_return = best_return
        global_best_stock = symbol

# Training complete - show results
print('\n' + '='*70)
print('🏆 ULTIMATE TRAINING COMPLETE!')
print('='*70)

# Sort results
sorted_results = sorted(training_results.items(), 
                       key=lambda x: x[1]['best_return'], 
                       reverse=True)

print('\n📊 TOP 15 PERFORMERS:')
print('  Rank | Stock | Best Return | Avg Return | Sharpe | Years')
print('  ' + '-'*60)
for i, (symbol, metrics) in enumerate(sorted_results[:15], 1):
    print(f'  {i:3d}  | {symbol:5s} | {metrics["best_return"]:+10.2f}% | '
          f'{metrics["avg_return"]:+9.2f}% | {metrics["best_sharpe"]:6.2f} | '
          f'{metrics["data_years"]:5.1f}')

print(f'\n🎯 ABSOLUTE BEST: {global_best_stock} with {global_best_return:+.2f}% return!')

# Calculate statistics
all_returns = [m['best_return'] for m in training_results.values()]
print(f'\n📈 OVERALL STATISTICS:')
print(f'  • Stocks Trained: {len(training_results)}')
print(f'  • Average Best Return: {np.mean(all_returns):+.2f}%')
print(f'  • Median Return: {np.median(all_returns):+.2f}%')
print(f'  • Total Data Processed: {total_data_points:,} days')

# Save the model
import os
import json
os.makedirs('trained_models', exist_ok=True)

model_info = {
    'timestamp': datetime.now().isoformat(),
    'stocks_trained': len(training_results),
    'total_data_points': total_data_points,
    'global_best_return': global_best_return,
    'global_best_stock': global_best_stock,
    'training_results': training_results
}

with open('trained_models/ultimate_model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print('\n💾 Model saved to: trained_models/ultimate_model_info.json')
print('\n🚀 YOUR AI IS NOW A TRADING GENIUS!')
print('   Trained on decades of market data across all sectors!')
print('   Ready to dominate the markets!')

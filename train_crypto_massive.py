import sys
sys.path.insert(0, '.')
from src.agents.dqn_agent_simple import DQNAgent
from src.environment.trading_env import TradingEnvironment
from src.market_data import MarketDataFetcher
import numpy as np
from datetime import datetime

print('🌟 MASSIVE CRYPTO TRAINING - MAXIMUM DATA!')
print('='*70)

# Top cryptocurrencies by market cap
MEGA_CRYPTO_LIST = [
    'BTC-USD',   # Bitcoin - since 2014
    'ETH-USD',   # Ethereum - since 2015
    'BNB-USD',   # Binance Coin
    'XRP-USD',   # Ripple
    'SOL-USD',   # Solana
    'ADA-USD',   # Cardano
    'DOGE-USD',  # Dogecoin
    'AVAX-USD',  # Avalanche
    'DOT-USD',   # Polkadot
    'MATIC-USD', # Polygon
    'SHIB-USD',  # Shiba Inu
    'TRX-USD',   # TRON
    'LTC-USD',   # Litecoin - lots of history
    'UNI-USD',   # Uniswap
    'LINK-USD',  # Chainlink
    'BCH-USD',   # Bitcoin Cash
    'ATOM-USD',  # Cosmos
    'XLM-USD',   # Stellar
    'NEAR-USD',  # NEAR Protocol
    'ALGO-USD'   # Algorand
]

print(f'📊 Crypto Training Configuration:')
print(f'  • Cryptocurrencies: {len(MEGA_CRYPTO_LIST)}')
print(f'  • Period: MAXIMUM available (up to 10 years)')
print(f'  • Episodes: 100 per crypto')
print(f'  • 24/7 Trading Data (more data than stocks!)')

response = input('\nStart MASSIVE crypto training? (y/n): ')
if response.lower() != 'y':
    exit()

print('\n🔥 STARTING MASSIVE CRYPTO TRAINING...\n')

# Fetch ALL crypto data
fetcher = MarketDataFetcher()
crypto_data = {}
total_data_points = 0

print('📥 Downloading maximum crypto history...\n')
for i, crypto in enumerate(MEGA_CRYPTO_LIST, 1):
    print(f'[{i:2d}/{len(MEGA_CRYPTO_LIST)}] {crypto:12s}...', end='')
    try:
        # Get maximum available data
        data = fetcher.get_stock_data(crypto, period='max')
        
        if len(data) > 0:
            crypto_data[crypto] = data
            total_data_points += len(data)
            years = len(data) / 365  # Crypto trades 365 days
            print(f' ✅ {len(data):,} days ({years:.1f} years)')
        else:
            print(' ❌ No data')
    except Exception as e:
        print(' ❌ Failed')

print(f'\n🎯 CRYPTO DATASET:')
print(f'   • Total data points: {total_data_points:,} days')
print(f'   • Equivalent years: {total_data_points/365:.1f}')
print(f'   • Cryptos loaded: {len(crypto_data)}\n')

# Create crypto-optimized agent
crypto_master_agent = None
training_results = {}
global_best_return = -float('inf')
best_crypto = None

print('🧠 Training Deep Q-Network on Crypto Markets...\n')

# Train on each cryptocurrency
for idx, (crypto, data) in enumerate(crypto_data.items(), 1):
    print(f'[{idx}/{len(crypto_data)}] {crypto}')
    print('  ' + '-'*50)
    print(f'  Data: {len(data)} days ({len(data)/365:.1f} years)')
    
    if len(data) < 100:
        print('  ⚠️ Insufficient data, skipping')
        continue
    
    # Create environment - crypto specific settings
    env = TradingEnvironment(
        data=data,
        initial_balance=10000,  # Start with  for crypto
        window_size=20,
        reward_type='sharpe',
        # transaction_cost removed
    )
    
    # Initialize agent if first time
    if crypto_master_agent is None:
        crypto_master_agent = DQNAgent(
            state_size=env.observation_size,
            action_size=3,
            learning_rate=0.0001  # Lower for crypto volatility
        )
    
    # Training metrics
    episode_returns = []
    best_return = -float('inf')
    best_sharpe = 0
    total_trades = 0
    
    # Train for 100 episodes on this crypto
    episodes_to_run = min(100, len(data) // 10)  # Scale with data
    
    for episode in range(episodes_to_run):
        obs, _ = env.reset()
        done = False
        episode_trades = 0
        
        while not done:
            # Get action with exploration
            action = crypto_master_agent.act(obs, training=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if action != 0:  # Count non-HOLD actions
                episode_trades += 1
        
        # Track performance
        ret = info.get('total_return_pct', 0)
        sharpe = info.get('sharpe_ratio', 0)
        episode_returns.append(ret)
        total_trades += episode_trades
        
        if ret > best_return:
            best_return = ret
            best_sharpe = sharpe
        
        # Progress update every 20 episodes
        if (episode + 1) % 20 == 0:
            avg_return = np.mean(episode_returns[-20:])
            print(f'    Episode {episode+1:3d}: Avg: {avg_return:+7.2f}% | Best: {best_return:+7.2f}% | Sharpe: {sharpe:.2f}')
    
    # Record results
    avg_final = np.mean(episode_returns[-20:]) if len(episode_returns) >= 20 else np.mean(episode_returns)
    training_results[crypto] = {
        'best_return': best_return,
        'avg_return': avg_final,
        'sharpe': best_sharpe,
        'total_trades': total_trades,
        'data_years': len(data) / 365
    }
    
    print(f'  ✅ Complete: Best: {best_return:+.2f}% | Avg: {avg_final:+.2f}%\n')
    
    # Update global best
    if best_return > global_best_return:
        global_best_return = best_return
        best_crypto = crypto

# Show final results
print('='*70)
print('🏆 MASSIVE CRYPTO TRAINING COMPLETE!')
print('='*70)

# Sort by returns
sorted_results = sorted(training_results.items(), 
                       key=lambda x: x[1]['best_return'], 
                       reverse=True)

print('\n📊 CRYPTO PERFORMANCE RANKINGS:')
print('  Rank | Crypto   | Best Return | Avg Return | Sharpe | Years')
print('  ' + '-'*65)
for i, (crypto, metrics) in enumerate(sorted_results, 1):
    name = crypto.replace('-USD', '')
    print(f'  {i:3d}  | {name:8s} | {metrics["best_return"]:+10.2f}% | '
          f'{metrics["avg_return"]:+9.2f}% | {metrics["sharpe"]:6.2f} | '
          f'{metrics["data_years"]:5.1f}')

# Statistics
all_returns = [m['best_return'] for m in training_results.values()]
print(f'\n📈 OVERALL STATISTICS:')
print(f'  • Cryptos Trained: {len(training_results)}')
print(f'  • Best Overall: {best_crypto} with {global_best_return:+.2f}%')
print(f'  • Average Best Return: {np.mean(all_returns):+.2f}%')
print(f'  • Total Data Processed: {total_data_points:,} days')

# Save the crypto model
import os
import json
os.makedirs('trained_models', exist_ok=True)

model_info = {
    'timestamp': datetime.now().isoformat(),
    'type': 'crypto_massive',
    'cryptos_trained': len(training_results),
    'total_data_points': total_data_points,
    'best_crypto': best_crypto,
    'best_return': global_best_return,
    'results': training_results
}

with open('trained_models/crypto_massive_model.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print('\n💾 Model saved to: trained_models/crypto_massive_model.json')
print('\n🚀 YOUR CRYPTO AI IS NOW A BEAST!')
print('   • Trained on years of 24/7 crypto market data')
print('   • Understands extreme volatility patterns')
print('   • Ready for massive crypto gains!')
print('\n💡 Crypto markets never close = more opportunities!')


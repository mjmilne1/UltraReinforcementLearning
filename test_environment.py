import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import environment
from src.environment.trading_env import TradingEnvironment, TradingAction

def create_sample_data(n_days=100):
    """Create sample OHLCV data"""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'high': prices * (1 + np.abs(np.random.uniform(0, 0.02, n_days))),
        'low': prices * (1 - np.abs(np.random.uniform(0, 0.02, n_days))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    return data

def test_environment():
    """Test the trading environment"""
    print("Testing Trading Environment")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data(252)  # One year of data
    print(f"Created {len(data)} days of sample data")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Create environment
    env = TradingEnvironment(
        data=data,
        initial_balance=100000,
        commission_rate=0.001,
        window_size=20,
        reward_type='sharpe'
    )
    print(f"\nEnvironment created:")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space.shape}")
    
    # Run one episode
    print("\n" + "="*60)
    print("Running test episode...")
    print("="*60)
    
    obs, info = env.reset()
    print(f"Initial state - Equity: ${info['current_equity']:.2f}")
    
    done = False
    step = 0
    actions_taken = {0: 0, 1: 0, 2: 0}
    
    while not done and step < 50:  # Limit to 50 steps for test
        # Random action for testing
        action = env.action_space.sample()
        actions_taken[action] += 1
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Print every 10 steps
        if step % 10 == 0:
            print(f"Step {step:3d}: Action={TradingAction.to_string(action)}, "
                  f"Reward={reward:7.4f}, Equity=${info['current_equity']:,.2f}, "
                  f"Return={info.get('total_return_pct', 0):6.2f}%")
        
        step += 1
    
    # Final results
    print("\n" + "="*60)
    print("Episode Summary:")
    print("="*60)
    final_metrics = env.portfolio.get_performance_metrics()
    
    print(f"Total Return: {final_metrics.get('total_return_pct', 0):.2f}%")
    print(f"Final Equity: ${final_metrics.get('current_equity', 0):,.2f}")
    print(f"Max Drawdown: {final_metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Total Trades: {final_metrics.get('total_trades', 0)}")
    print(f"\nActions taken: HOLD={actions_taken[0]}, BUY={actions_taken[1]}, SELL={actions_taken[2]}")
    
    print("\n✅ Trading environment test complete!")

if __name__ == '__main__':
    test_environment()

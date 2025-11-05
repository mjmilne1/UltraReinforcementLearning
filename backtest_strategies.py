import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List

# Import our components
from src.market_data import MarketDataFetcher
from src.environment.trading_env import TradingEnvironment
from src.agents.dqn_agent_simple import DQNAgent
from src.strategies.ml_agents.ppo_agent import PPOAgent
from src.strategies.ml_agents.a2c_agent import A2CAgent
from src.strategies.classical.traditional_strategies import (
    MeanReversionStrategy, MomentumStrategy
)

class StrategyBacktester:
    '''Backtest multiple trading strategies'''
    
    def __init__(self, symbols: List[str], period: str = '1y'):
        self.symbols = symbols
        self.period = period
        self.results = {}
        
        # Fetch data
        print(f'Fetching {period} of data for {", ".join(symbols)}...')
        self.fetcher = MarketDataFetcher()
        self.data = {}
        for symbol in symbols:
            self.data[symbol] = self.fetcher.get_stock_data(symbol, period)
    
    def backtest_ml_strategy(self, strategy_name: str, agent, symbol: str):
        '''Backtest an ML strategy'''
        print(f'\nTesting {strategy_name} on {symbol}...')
        
        # Create environment
        env = TradingEnvironment(
            data=self.data[symbol],
            initial_balance=100000,
            window_size=20,
            reward_type='sharpe'
        )
        
        # Run episode
        obs, info = env.reset()
        done = False
        
        while not done:
            action = agent.act(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        return {
            'final_value': info['current_equity'],
            'total_return': info['total_return_pct'],
            'sharpe_ratio': info.get('sharpe_ratio', 0),
            'max_drawdown': info.get('max_drawdown_pct', 0),
            'total_trades': info.get('total_trades', 0),
            'win_rate': info.get('win_rate', 0)
        }
    
    def backtest_classical_strategy(self, strategy_name: str, strategy, symbol: str):
        '''Backtest a classical strategy'''
        print(f'\nTesting {strategy_name} on {symbol}...')
        
        data = self.data[symbol]
        prices = data['close'].values
        
        # Simulate trading
        cash = 100000
        position = 0
        trades = 0
        equity_curve = [cash]
        
        for i in range(strategy.lookback if hasattr(strategy, 'lookback') else 20, len(prices)):
            price_window = prices[:i+1]
            signal = strategy.generate_signal(price_window)
            
            if signal == 1 and cash > prices[i]:  # Buy
                position = cash / prices[i]
                cash = 0
                trades += 1
            elif signal == -1 and position > 0:  # Sell
                cash = position * prices[i]
                position = 0
                trades += 1
            
            # Update equity
            equity = cash + position * prices[i] if position > 0 else cash
            equity_curve.append(equity)
        
        # Calculate metrics
        returns = np.diff(equity_curve) / equity_curve[:-1]
        total_return = (equity_curve[-1] - 100000) / 100000 * 100
        
        sharpe = 0
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        max_dd = 0
        if len(equity_curve) > 0:
            peaks = np.maximum.accumulate(equity_curve)
            dd = (np.array(equity_curve) - peaks) / peaks
            max_dd = np.min(dd) * 100
        
        return {
            'final_value': equity_curve[-1] if equity_curve else 100000,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': trades,
            'win_rate': 0.5  # Simplified
        }
    
    def run_all_backtests(self):
        '''Run all strategy backtests'''
        print('\n' + '='*60)
        print('BACKTESTING ALL STRATEGIES')
        print('='*60)
        
        all_results = {}
        
        for symbol in self.symbols:
            print(f'\n### Testing on {symbol} ###')
            symbol_results = {}
            
            # Test DQN
            try:
                dqn = DQNAgent(state_size=40, action_size=3)
                symbol_results['DQN'] = self.backtest_ml_strategy('DQN', dqn, symbol)
            except Exception as e:
                print(f'DQN error: {e}')
            
            # Test PPO
            try:
                ppo = PPOAgent(state_size=40, action_size=3)
                symbol_results['PPO'] = self.backtest_ml_strategy('PPO', ppo, symbol)
            except Exception as e:
                print(f'PPO error: {e}')
            
            # Test A2C
            try:
                a2c = A2CAgent(state_size=40, action_size=3)
                symbol_results['A2C'] = self.backtest_ml_strategy('A2C', a2c, symbol)
            except Exception as e:
                print(f'A2C error: {e}')
            
            # Test Mean Reversion
            try:
                mr = MeanReversionStrategy()
                symbol_results['Mean Reversion'] = self.backtest_classical_strategy(
                    'Mean Reversion', mr, symbol
                )
            except Exception as e:
                print(f'Mean Reversion error: {e}')
            
            # Test Momentum
            try:
                mom = MomentumStrategy()
                symbol_results['Momentum'] = self.backtest_classical_strategy(
                    'Momentum', mom, symbol
                )
            except Exception as e:
                print(f'Momentum error: {e}')
            
            all_results[symbol] = symbol_results
        
        self.results = all_results
        return all_results
    
    def print_results(self):
        '''Print formatted results'''
        print('\n' + '='*60)
        print('BACKTEST RESULTS SUMMARY')
        print('='*60)
        
        for symbol, strategies in self.results.items():
            print(f'\n{symbol} Results:')
            print('-'*40)
            
            # Sort by return
            sorted_strats = sorted(
                strategies.items(), 
                key=lambda x: x[1]['total_return'], 
                reverse=True
            )
            
            for name, metrics in sorted_strats:
                print(f'\n  {name}:')
                print(f'    Return:      {metrics["total_return"]:+.2f}%')
                print(f'    Sharpe:      {metrics["sharpe_ratio"]:.2f}')
                print(f'    Max DD:      {metrics["max_drawdown"]:.2f}%')
                print(f'    Final Value: ')
                print(f'    Trades:      {metrics["total_trades"]}')
        
        # Overall winner
        print('\n' + '='*60)
        print('OVERALL BEST STRATEGY')
        print('='*60)
        
        best_return = -float('inf')
        best_strategy = None
        best_symbol = None
        
        for symbol, strategies in self.results.items():
            for name, metrics in strategies.items():
                if metrics['total_return'] > best_return:
                    best_return = metrics['total_return']
                    best_strategy = name
                    best_symbol = symbol
        
        if best_strategy:
            print(f'\n🏆 Winner: {best_strategy} on {best_symbol}')
            print(f'   Return: {best_return:+.2f}%')

# Run backtests
if __name__ == '__main__':
    print('🚀 ULTRA RL - Strategy Backtesting System')
    print('='*50)
    
    # Test on multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    backtester = StrategyBacktester(symbols, period='6mo')
    backtester.run_all_backtests()
    backtester.print_results()
    
    print('\n✅ Backtesting complete!')

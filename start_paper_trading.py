import sys
sys.path.insert(0, '.')
from src.paper_trading.paper_trading_engine import PaperTradingAccount, PaperTradingEngine
from src.strategies.classical.traditional_strategies import MomentumStrategy

print('💰 ULTRA RL - Paper Trading System')
print('='*50)

# Create or load account
account = PaperTradingAccount(
    starting_capital=100000,
    account_name='MainAccount'
)

# Show account status
print('\n📊 Account Status:')
print(f'   Starting Capital: ')
print(f'   Current Cash: ')
print(f'   Positions: {len(account.positions)}')

# Select strategy
print('\n🎯 Available Strategies:')
print('1. Momentum (57% backtest return)')
print('2. Mean Reversion (25% return)')
print('3. DQN AI Agent')
print('4. Ensemble (all strategies)')

choice = input('\nSelect strategy (1-4): ')

# Trading symbols
symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']

# Initialize strategy
if choice == '1':
    strategy = MomentumStrategy()
    print('✅ Using Momentum Strategy')
else:
    strategy = MomentumStrategy()  # Default for now
    print('✅ Using Momentum Strategy (default)')

# Create trading engine
engine = PaperTradingEngine(strategy, symbols, account)

print(f'\n🎯 Trading: {", ".join(symbols)}')
print('\n⚡ Starting paper trading...')
print('   (Press Ctrl+C to stop)\n')

# Run paper trading
try:
    engine.run_trading_session(duration_hours=0.1)  # 6 minutes for demo
except KeyboardInterrupt:
    print('\n⛔ Trading stopped by user')
    engine.print_performance_report()

print('\n✅ Paper trading session complete!')
print('   Account data saved to: paper_trading_data/')

import sys
sys.path.insert(0, '.')
from src.portfolio_optimization import PortfolioOptimization
from src.market_data import MarketDataFetcher
from src.agents.dqn_agent_simple import DQNAgent
from src.environment.trading_env import TradingEnvironment

print('🚀 ULTRA RL - Master Trading System')
print('='*50)

# 1. Get optimal portfolio
symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
optimizer = PortfolioOptimization(symbols)
optimizer.fetch_historical_data('3mo')
optimal = optimizer.optimize_portfolio()

print('Optimal Portfolio Allocation:')
for sym, weight in optimal['weights'].items():
    print(f'  {sym}: {weight*100:.1f}%')
print(f'Expected Sharpe: {optimal["metrics"]["sharpe"]:.2f}')

# 2. Fetch MORE market data (3 months instead of 1 month)
fetcher = MarketDataFetcher()
data = fetcher.get_stock_data(symbols[0], '3mo', '1d')  # Changed to 3mo

if len(data) < 30:
    print(f'Warning: Only {len(data)} days of data. Fetching more...')
    data = fetcher.get_stock_data(symbols[0], '6mo', '1d')

# 3. Run AI trading
env = TradingEnvironment(data, initial_balance=100000)
agent = DQNAgent(state_size=env.observation_size, action_size=3)

print(f'\n🤖 AI Trading Active')
print(f'💰 Starting Capital: ,000')
print(f'📊 Data points: {len(data)}')

# Quick training
print('\nTraining agent...')
obs, info = env.reset()
for step in range(100):
    action = agent.act(obs, training=True)
    next_obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
    obs = next_obs

metrics = env.portfolio.get_performance_metrics()
print(f'\n✅ System FULLY OPERATIONAL!')
print(f'📈 Current Return: {metrics["total_return_pct"]:.2f}%')
print(f'💼 Portfolio Value: ')

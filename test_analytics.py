import sys
sys.path.insert(0, '.')
import numpy as np
import json
from src.analytics.advanced_analytics import AdvancedAnalytics

print('📊 ADVANCED ANALYTICS SYSTEM')
print('='*60)

# Initialize analytics
analytics = AdvancedAnalytics(risk_free_rate=0.05)

# Generate sample returns
np.random.seed(42)
sample_returns = np.random.normal(0.001, 0.02, 252)

print('\n1. PERFORMANCE METRICS')
print('-'*40)
sharpe = analytics.calculate_sharpe_ratio(sample_returns)
sortino = analytics.calculate_sortino_ratio(sample_returns)
print('  Sharpe Ratio: {:.2f}'.format(sharpe))
print('  Sortino Ratio: {:.2f}'.format(sortino))

print('\n2. RISK ANALYTICS')
print('-'*40)
risk = analytics.risk_metrics(sample_returns)
print('  Annual Volatility: {:.2%}'.format(risk['volatility']))
print('  Value at Risk (95%): {:.2%}'.format(risk['var_95']))
print('  Skewness: {:.2f}'.format(risk['skewness']))

print('\n3. MONTE CARLO SIMULATION')
print('-'*40)
mc = analytics.monte_carlo_simulation(sample_returns, initial_capital=100000)
print('  Expected Value (1Y): '.format(mc['expected_value']))
print('  Probability of Profit: {:.1f}%'.format(mc['probability_profit']))

print('\n4. DRAWDOWN ANALYSIS')
print('-'*40)
equity_curve = 100000 * np.cumprod(1 + sample_returns)
dd = analytics.calculate_max_drawdown(equity_curve)
print('  Max Drawdown: {:.2f}%'.format(dd['max_drawdown']))
print('  Time Underwater: {:.1f}%'.format(dd['time_underwater']))

print('\n✅ Advanced Analytics Ready!')
print('💡 These are institutional-grade metrics!')

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import time
from src.indicators.technical_indicators import TechnicalIndicators
from src.indicators.risk_metrics import RiskCalculator, RiskMethod

def test_integrated_analysis():
    '''Test integrated technical and risk analysis'''
    
    print('Integrated Technical & Risk Analysis')
    print('=' * 60)
    
    # Generate realistic market data
    np.random.seed(42)
    n = 252  # One year of daily data
    
    # Simulate trending market with volatility
    trend = 0.0008  # Slight upward trend
    volatility = 0.015
    returns = np.random.normal(trend, volatility, n)
    prices = 100 * (1 + returns).cumprod()
    
    # Add high/low/volume
    high = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volume = np.random.randint(1000000, 5000000, n).astype(float)
    
    # Benchmark (market index)
    benchmark_returns = returns + np.random.normal(0, 0.003, n)
    benchmark_prices = 100 * (1 + benchmark_returns).cumprod()
    
    # Technical Analysis
    print('\n1. TECHNICAL INDICATORS')
    print('-' * 40)
    ti = TechnicalIndicators()
    indicators = ti.calculate_all(prices, high, low, volume)
    
    for name, values in list(indicators.items())[:5]:
        if not np.all(np.isnan(values)):
            last = values[~np.isnan(values)][-1]
            print(f'  {name:15} {last:10.2f}')
    
    # Risk Analysis
    print('\n2. RISK METRICS')
    print('-' * 40)
    risk_calc = RiskCalculator()
    metrics = risk_calc.calculate_all_metrics(prices, benchmark_prices)
    
    print(f'  Sharpe Ratio:    {metrics.sharpe_ratio:8.3f}')
    print(f'  Sortino Ratio:   {metrics.sortino_ratio:8.3f}')
    print(f'  Max Drawdown:    {metrics.max_drawdown:8.1%}')
    print(f'  VaR (95%):       {metrics.var_95:8.1%}')
    print(f'  Beta:            {metrics.beta:8.3f}')
    
    # Performance Summary
    print('\n3. PERFORMANCE SUMMARY')
    print('-' * 40)
    annual_return = (prices[-1] / prices[0]) ** (252/n) - 1
    print(f'  Annual Return:   {annual_return:8.1%}')
    print(f'  Annual Vol:      {metrics.volatility:8.1%}')
    print(f'  Risk/Reward:     {annual_return/metrics.volatility:8.2f}')
    
    print('\n✅ Integrated analysis complete!')

if __name__ == '__main__':
    test_integrated_analysis()

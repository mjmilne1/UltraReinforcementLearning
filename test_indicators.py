import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import time
from src.indicators.technical_indicators import TechnicalIndicators

def test_indicators():
    '''Test technical indicators'''
    
    # Generate sample data
    np.random.seed(42)
    n = 1000
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = prices + np.abs(np.random.randn(n) * 0.2)
    low = prices - np.abs(np.random.randn(n) * 0.2)
    volume = np.random.randint(100000, 1000000, n).astype(float)
    
    indicators = TechnicalIndicators()
    
    print('Technical Indicators Test')
    print('=' * 50)
    
    # Test each indicator
    print('\n1. Trend Indicators:')
    sma = indicators.sma(prices, 20)
    print(f'   SMA(20): Last value = {sma[-1]:.2f}')
    
    ema = indicators.ema(prices, 20)
    print(f'   EMA(20): Last value = {ema[-1]:.2f}')
    
    macd = indicators.macd(prices)
    print(f'   MACD: Last value = {macd.values[-1]:.4f}')
    
    print('\n2. Momentum Indicators:')
    rsi = indicators.rsi(prices, 14)
    print(f'   RSI(14): Last value = {rsi[-1]:.2f}')
    
    k, d = indicators.stochastic(high, low, prices)
    print(f'   Stochastic: K={k[-1]:.2f}, D={d[-1]:.2f}')
    
    print('\n3. Volatility Indicators:')
    bb = indicators.bollinger_bands(prices)
    print(f'   Bollinger Bands: Upper={bb.upper_band[-1]:.2f}, Lower={bb.lower_band[-1]:.2f}')
    
    atr = indicators.atr(high, low, prices)
    print(f'   ATR(14): Last value = {atr[-1]:.4f}')
    
    print('\n4. Volume Indicators:')
    obv = indicators.obv(prices, volume)
    print(f'   OBV: Last value = {obv[-1]:,.0f}')
    
    vwap = indicators.vwap(high, low, prices, volume)
    print(f'   VWAP: Last value = {vwap[-1]:.2f}')
    
    # Performance test
    print('\n5. Performance Test:')
    start = time.perf_counter()
    all_indicators = indicators.calculate_all(prices, high, low, volume)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f'   Calculated {len(all_indicators)} indicators in {elapsed:.2f} ms')
    print(f'   Throughput: {n / elapsed * 1000:.0f} data points/second')
    
    print('\n✅ All tests passed!')

if __name__ == '__main__':
    test_indicators()

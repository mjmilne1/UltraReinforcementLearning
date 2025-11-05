'''High-Performance Technical Indicators Engine'''
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class IndicatorResult:
    name: str
    values: np.ndarray
    signal: Optional[np.ndarray] = None
    upper_band: Optional[np.ndarray] = None
    lower_band: Optional[np.ndarray] = None

class TechnicalIndicators:
    '''High-performance technical indicators'''
    
    @staticmethod
    def sma(prices: np.ndarray, period: int) -> np.ndarray:
        '''Simple Moving Average'''
        n = len(prices)
        sma = np.full(n, np.nan)
        for i in range(period-1, n):
            sma[i] = np.mean(prices[i-period+1:i+1])
        return sma
    
    @staticmethod
    def ema(prices: np.ndarray, period: int) -> np.ndarray:
        '''Exponential Moving Average'''
        ema = np.full_like(prices, np.nan)
        ema[period-1] = np.mean(prices[:period])
        k = 2.0 / (period + 1)
        for i in range(period, len(prices)):
            ema[i] = prices[i] * k + ema[i-1] * (1 - k)
        return ema    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        '''Relative Strength Index'''
        n = len(prices)
        rsi = np.full(n, np.nan)
        
        if n < period + 1:
            return rsi
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        for i in range(period, n-1):
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi[i+1] = 100 - (100 / (1 + rs))
            else:
                rsi[i+1] = 100
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        return rsi
    
    def macd(self, prices: np.ndarray, fast=12, slow=26, signal=9):
        '''MACD indicator'''
        fast_ema = self.ema(prices, fast)
        slow_ema = self.ema(prices, slow)
        macd_line = fast_ema - slow_ema
        signal_line = self.ema(macd_line[~np.isnan(macd_line)], signal)
        
        signal_aligned = np.full_like(macd_line, np.nan)
        signal_aligned[-len(signal_line):] = signal_line
        histogram = macd_line - signal_aligned
        
        return IndicatorResult(
            name='MACD',
            values=macd_line,
            signal=signal_aligned
        )    
    def bollinger_bands(self, prices: np.ndarray, period=20, std_dev=2):
        '''Bollinger Bands'''
        middle = self.sma(prices, period)
        std = np.array([np.std(prices[max(0,i-period+1):i+1]) if i >= period-1 else np.nan 
                       for i in range(len(prices))])
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return IndicatorResult(
            name='Bollinger Bands',
            values=middle,
            upper_band=upper,
            lower_band=lower
        )
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14):
        '''Average True Range'''
        n = len(close)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        atr = np.full(n, np.nan)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr
    
    @staticmethod
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        '''On-Balance Volume'''
        obv = np.zeros(len(close))
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return obv    
    @staticmethod
    def vwap(high, low, close, volume):
        '''Volume Weighted Average Price'''
        typical_price = (high + low + close) / 3
        return np.cumsum(typical_price * volume) / np.cumsum(volume)
    
    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        '''Stochastic Oscillator'''
        n = len(close)
        k_values = np.full(n, np.nan)
        
        for i in range(k_period - 1, n):
            period_high = np.max(high[i - k_period + 1:i + 1])
            period_low = np.min(low[i - k_period + 1:i + 1])
            if period_high != period_low:
                k_values[i] = ((close[i] - period_low) / (period_high - period_low)) * 100
            else:
                k_values[i] = 50
        
        d_values = np.full(n, np.nan)
        for i in range(k_period + d_period - 2, n):
            d_values[i] = np.mean(k_values[i - d_period + 1:i + 1])
        
        return k_values, d_values
    
    def calculate_all(self, prices, high=None, low=None, volume=None):
        '''Calculate all indicators'''
        results = {}
        
        results['SMA_20'] = self.sma(prices, 20)
        results['EMA_20'] = self.ema(prices, 20)
        results['RSI_14'] = self.rsi(prices, 14)
        
        macd = self.macd(prices)
        results['MACD'] = macd.values
        results['MACD_Signal'] = macd.signal
        
        bb = self.bollinger_bands(prices)
        results['BB_Middle'] = bb.values
        results['BB_Upper'] = bb.upper_band
        results['BB_Lower'] = bb.lower_band
        
        if high is not None and low is not None:
            results['ATR'] = self.atr(high, low, prices)
            k, d = self.stochastic(high, low, prices)
            results['Stoch_K'] = k
            results['Stoch_D'] = d
            
            if volume is not None:
                results['VWAP'] = self.vwap(high, low, prices, volume)
                results['OBV'] = self.obv(prices, volume)
        
        return results

def test():
    '''Test the indicators'''
    np.random.seed(42)
    n = 1000
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = prices + np.abs(np.random.randn(n) * 0.2)
    low = prices - np.abs(np.random.randn(n) * 0.2)
    volume = np.random.randint(100000, 1000000, n).astype(float)
    
    ti = TechnicalIndicators()
    results = ti.calculate_all(prices, high, low, volume)
    
    print('Technical Indicators Test Results:')
    print('=' * 50)
    for name, values in results.items():
        if not np.all(np.isnan(values)):
            last_valid = values[~np.isnan(values)][-1]
            print(f'{name:15} Last Value: {last_valid:10.2f}')
    print('\nAll indicators calculated successfully!')

if __name__ == '__main__':
    test()

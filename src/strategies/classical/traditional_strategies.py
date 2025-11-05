import numpy as np
import pandas as pd
from typing import Dict, Optional

class MeanReversionStrategy:
    '''Mean Reversion Trading Strategy'''
    
    def __init__(self, lookback: int = 20, entry_threshold: float = 2.0, exit_threshold: float = 0.5):
        self.lookback = lookback
        self.entry_threshold = entry_threshold  # Z-score for entry
        self.exit_threshold = exit_threshold    # Z-score for exit
        self.position = 0
        
    def calculate_zscore(self, prices: np.ndarray) -> float:
        '''Calculate z-score of current price'''
        if len(prices) < self.lookback:
            return 0.0
        
        recent_prices = prices[-self.lookback:]
        mean = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        if std == 0:
            return 0.0
            
        zscore = (prices[-1] - mean) / std
        return zscore
    
    def generate_signal(self, prices: np.ndarray) -> int:
        '''Generate trading signal: 1=buy, -1=sell, 0=hold'''
        zscore = self.calculate_zscore(prices)
        
        # Entry signals
        if zscore < -self.entry_threshold and self.position <= 0:
            self.position = 1
            return 1  # Buy (oversold)
        elif zscore > self.entry_threshold and self.position >= 0:
            self.position = -1
            return -1  # Sell (overbought)
        
        # Exit signals
        elif abs(zscore) < self.exit_threshold and self.position != 0:
            signal = -self.position  # Close position
            self.position = 0
            return signal
        
        return 0  # Hold

class MomentumStrategy:
    '''Momentum Trading Strategy'''
    
    def __init__(self, short_window: int = 10, long_window: int = 30):
        self.short_window = short_window
        self.long_window = long_window
        self.position = 0
        
    def calculate_momentum(self, prices: np.ndarray) -> Dict:
        '''Calculate momentum indicators'''
        if len(prices) < self.long_window:
            return {'signal': 0}
        
        # Simple moving averages
        sma_short = np.mean(prices[-self.short_window:])
        sma_long = np.mean(prices[-self.long_window:])
        
        # Rate of change
        roc = (prices[-1] - prices[-self.short_window]) / prices[-self.short_window]
        
        # MACD-like signal
        macd = sma_short - sma_long
        signal_line = np.mean(prices[-9:]) - np.mean(prices[-26:]) if len(prices) >= 26 else 0
        
        return {
            'sma_short': sma_short,
            'sma_long': sma_long,
            'roc': roc,
            'macd': macd,
            'signal_line': signal_line
        }
    
    def generate_signal(self, prices: np.ndarray) -> int:
        '''Generate trading signal'''
        indicators = self.calculate_momentum(prices)
        
        if len(prices) < self.long_window:
            return 0
        
        # Bullish signals
        if (indicators['sma_short'] > indicators['sma_long'] and 
            indicators['roc'] > 0.02 and 
            indicators['macd'] > indicators['signal_line']):
            if self.position <= 0:
                self.position = 1
                return 1  # Buy
        
        # Bearish signals
        elif (indicators['sma_short'] < indicators['sma_long'] and 
              indicators['roc'] < -0.02 and 
              indicators['macd'] < indicators['signal_line']):
            if self.position >= 0:
                self.position = -1
                return -1  # Sell
        
        return 0  # Hold

class PairsTradingStrategy:
    '''Statistical Arbitrage - Pairs Trading'''
    
    def __init__(self, lookback: int = 60, entry_zscore: float = 2.0, exit_zscore: float = 0.5):
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.position = 0
        self.hedge_ratio = None
        
    def calculate_spread(self, prices1: np.ndarray, prices2: np.ndarray) -> np.ndarray:
        '''Calculate spread between two assets'''
        if len(prices1) != len(prices2):
            raise ValueError('Price arrays must have same length')
        
        # Calculate hedge ratio using OLS
        from scipy import stats
        slope, intercept, _, _, _ = stats.linregress(prices2, prices1)
        self.hedge_ratio = slope
        
        # Calculate spread
        spread = prices1 - slope * prices2
        return spread
    
    def generate_signals(self, prices1: np.ndarray, prices2: np.ndarray) -> Dict:
        '''Generate pairs trading signals'''
        if len(prices1) < self.lookback or len(prices2) < self.lookback:
            return {'asset1': 0, 'asset2': 0}
        
        spread = self.calculate_spread(prices1[-self.lookback:], prices2[-self.lookback:])
        
        # Calculate z-score
        mean_spread = np.mean(spread)
        std_spread = np.std(spread)
        
        if std_spread == 0:
            return {'asset1': 0, 'asset2': 0}
        
        current_zscore = (spread[-1] - mean_spread) / std_spread
        
        # Generate signals
        if current_zscore > self.entry_zscore and self.position <= 0:
            # Spread too high - short asset1, long asset2
            self.position = -1
            return {'asset1': -1, 'asset2': 1}
        elif current_zscore < -self.entry_zscore and self.position >= 0:
            # Spread too low - long asset1, short asset2
            self.position = 1
            return {'asset1': 1, 'asset2': -1}
        elif abs(current_zscore) < self.exit_zscore and self.position != 0:
            # Close positions
            signals = {'asset1': -self.position, 'asset2': self.position}
            self.position = 0
            return signals
        
        return {'asset1': 0, 'asset2': 0}


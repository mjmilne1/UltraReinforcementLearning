"""Trading Environment for RL Agents
Gym-compatible environment for training trading agents
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.environment.portfolio.portfolio_manager import Portfolio
from src.indicators.technical_indicators import TechnicalIndicators
from src.indicators.risk_metrics import RiskCalculator

class TradingAction:
    """Trading actions"""
    HOLD = 0
    BUY = 1
    SELL = 2
    
    @staticmethod
    def to_string(action: int) -> str:
        return {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(action, 'UNKNOWN')

class TradingEnvironment(gym.Env):
    """Gym-compatible trading environment"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                 data: pd.DataFrame,
                 initial_balance: float = 100000,
                 commission_rate: float = 0.001,
                 window_size: int = 50,
                 reward_type: str = 'sharpe',
                 position_sizing: str = 'fixed',
                 max_position_size: float = 0.1):
        """
        Initialize trading environment
        
        Args:
            data: DataFrame with OHLCV data
            initial_balance: Starting capital
            commission_rate: Commission per trade
            window_size: Lookback window for observations
            reward_type: 'profit', 'sharpe', 'risk_adjusted'
            position_sizing: 'fixed', 'kelly', 'risk_parity'
            max_position_size: Maximum position size as fraction of portfolio
        """
        super(TradingEnvironment, self).__init__()
        
        # Market data
        self.data = data
        self.prices = data['close'].values
        self.window_size = window_size
        
        # Portfolio settings
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.position_sizing = position_sizing
        self.max_position_size = max_position_size
        
        # Reward settings
        self.reward_type = reward_type
        
        # Technical indicators
        self.indicators = TechnicalIndicators()
        self.risk_calc = RiskCalculator()
        
        # Calculate indicators for the entire dataset
        self._precompute_indicators()
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
        
        # Observation: price features + indicators + portfolio state
        self.observation_size = self._calculate_observation_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.observation_size,), 
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def _precompute_indicators(self):
        """Pre-calculate technical indicators"""
        high = self.data['high'].values if 'high' in self.data else self.prices
        low = self.data['low'].values if 'low' in self.data else self.prices
        volume = self.data['volume'].values if 'volume' in self.data else np.ones_like(self.prices)
        
        # Calculate all indicators
        self.indicator_data = self.indicators.calculate_all(
            self.prices, high, low, volume
        )
    
    def _calculate_observation_size(self) -> int:
        """Calculate the size of observation vector"""
        # Price features (returns for window)
        price_features = self.window_size
        
        # Technical indicators (assuming ~15 indicators)
        indicator_features = 15
        
        # Portfolio features (cash_pct, position_pct, unrealized_pnl, etc.)
        portfolio_features = 5
        
        return price_features + indicator_features + portfolio_features
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset portfolio
        self.portfolio = Portfolio(
            initial_cash=self.initial_balance,
            commission_rate=self.commission_rate
        )
        
        # Reset episode tracking
        self.current_step = self.window_size
        self.done = False
        
        # Performance tracking
        self.episode_trades = 0
        self.episode_return = 0.0
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Trading action (HOLD=0, BUY=1, SELL=2)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Store previous equity
        prev_equity = self.portfolio.equity
        
        # Execute action
        self._execute_action(action)
        
        # Update portfolio with current prices
        current_price = self.prices[self.current_step]
        self.portfolio.update_prices({'asset': current_price})
        
        # Calculate reward
        reward = self._calculate_reward(prev_equity)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.prices) - 1
        truncated = self.portfolio.equity <= self.initial_balance * 0.5  # 50% loss
        
        # Get new observation
        obs = self._get_observation()
        info = self._get_info()
        info['action'] = TradingAction.to_string(action)
        
        return obs, reward, terminated, truncated, info
    
    def _execute_action(self, action: int):
        """Execute trading action"""
        current_price = self.prices[self.current_step]
        position_size = self._calculate_position_size()
        
        if action == TradingAction.BUY:
            # Buy if we have cash
            quantity = position_size / current_price
            self.portfolio.execute_buy('asset', quantity, current_price)
            self.episode_trades += 1
            
        elif action == TradingAction.SELL:
            # Sell if we have position
            if 'asset' in self.portfolio.positions:
                position = self.portfolio.positions['asset']
                self.portfolio.execute_sell('asset', position.quantity, current_price)
                self.episode_trades += 1
    
    def _calculate_position_size(self) -> float:
        """Calculate position size based on strategy"""
        if self.position_sizing == 'fixed':
            return self.portfolio.equity * self.max_position_size
        elif self.position_sizing == 'kelly':
            # Simplified Kelly criterion
            return self._kelly_position_size()
        elif self.position_sizing == 'risk_parity':
            return self._risk_parity_position_size()
        return self.portfolio.equity * self.max_position_size
    
    def _kelly_position_size(self) -> float:
        """Kelly criterion position sizing"""
        # Simplified: use recent returns to estimate win probability and payoff
        if self.current_step < 20:
            return self.portfolio.equity * 0.05
        
        recent_returns = np.diff(self.prices[self.current_step-20:self.current_step]) / \
                        self.prices[self.current_step-20:self.current_step-1]
        
        win_rate = np.mean(recent_returns > 0)
        avg_win = np.mean(recent_returns[recent_returns > 0]) if np.any(recent_returns > 0) else 0
        avg_loss = abs(np.mean(recent_returns[recent_returns < 0])) if np.any(recent_returns < 0) else 1
        
        if avg_loss > 0:
            kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_pct = np.clip(kelly_pct, 0, self.max_position_size)
        else:
            kelly_pct = self.max_position_size
        
        return self.portfolio.equity * kelly_pct
    
    def _risk_parity_position_size(self) -> float:
        """Risk parity position sizing"""
        # Use volatility to size position
        if self.current_step < 20:
            return self.portfolio.equity * 0.05
        
        recent_returns = np.diff(self.prices[self.current_step-20:self.current_step]) / \
                        self.prices[self.current_step-20:self.current_step-1]
        volatility = np.std(recent_returns)
        
        if volatility > 0:
            # Inverse volatility weighting
            target_vol = 0.02  # 2% daily volatility target
            position_pct = min(target_vol / volatility, self.max_position_size)
        else:
            position_pct = self.max_position_size
        
        return self.portfolio.equity * position_pct
    
    def _calculate_reward(self, prev_equity: float) -> float:
        """Calculate reward based on strategy"""
        current_equity = self.portfolio.equity
        
        if self.reward_type == 'profit':
            # Simple profit-based reward
            return (current_equity - prev_equity) / self.initial_balance
            
        elif self.reward_type == 'sharpe':
            # Sharpe ratio-based reward
            if len(self.portfolio.returns) > 1:
                returns = np.array(self.portfolio.returns[-20:])  # Last 20 returns
                if np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns)
                    return sharpe / 10  # Scale down
                else:
                    return 0
            return 0
            
        elif self.reward_type == 'risk_adjusted':
            # Risk-adjusted return
            ret = (current_equity - prev_equity) / prev_equity
            
            # Penalize drawdown
            drawdown_penalty = 0
            if current_equity < self.portfolio.peak_equity:
                drawdown = (self.portfolio.peak_equity - current_equity) / self.portfolio.peak_equity
                drawdown_penalty = -drawdown * 2  # 2x penalty for drawdowns
            
            return ret + drawdown_penalty
        
        return 0
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = []
        
        # Price returns for window
        if self.current_step >= self.window_size:
            price_window = self.prices[self.current_step-self.window_size:self.current_step]
            returns = np.diff(price_window) / price_window[:-1]
            obs.extend(returns)
        else:
            obs.extend([0] * (self.window_size - 1))
        
        # Technical indicators (normalized)
        for indicator_name, values in list(self.indicator_data.items())[:15]:
            if self.current_step < len(values) and not np.isnan(values[self.current_step]):
                # Normalize using z-score
                val = values[self.current_step]
                recent_values = values[max(0, self.current_step-50):self.current_step+1]
                recent_values = recent_values[~np.isnan(recent_values)]
                if len(recent_values) > 1:
                    mean = np.mean(recent_values)
                    std = np.std(recent_values)
                    normalized = (val - mean) / (std + 1e-8)
                    obs.append(np.clip(normalized, -3, 3))
                else:
                    obs.append(0)
            else:
                obs.append(0)
        
        # Portfolio features
        equity_pct_change = (self.portfolio.equity - self.initial_balance) / self.initial_balance
        cash_pct = self.portfolio.cash / self.portfolio.equity if self.portfolio.equity > 0 else 1
        position_pct = self.portfolio.exposure / self.portfolio.equity if self.portfolio.equity > 0 else 0
        
        # Unrealized P&L
        unrealized_pnl = 0
        if 'asset' in self.portfolio.positions:
            unrealized_pnl = self.portfolio.positions['asset'].unrealized_pnl_pct
        
        # Drawdown
        drawdown = 0
        if self.portfolio.peak_equity > 0:
            drawdown = (self.portfolio.equity - self.portfolio.peak_equity) / self.portfolio.peak_equity
        
        obs.extend([
            equity_pct_change,
            cash_pct,
            position_pct,
            unrealized_pnl,
            drawdown
        ])
        
        # Ensure correct size
        obs = obs[:self.observation_size]
        while len(obs) < self.observation_size:
            obs.append(0)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info"""
        metrics = self.portfolio.get_performance_metrics()
        metrics['step'] = self.current_step
        metrics['episode_trades'] = self.episode_trades
        return metrics
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            metrics = self.portfolio.get_performance_metrics()
            print(f"Step: {self.current_step}/{len(self.prices)}")
            print(f"Equity: ${metrics.get('current_equity', 0):.2f}")
            print(f"Return: {metrics.get('total_return_pct', 0):.2f}%")
            print(f"Positions: {metrics.get('current_positions', 0)}")
            print("-" * 40)

'''Risk Metrics Calculation System
Ultra RL Platform - Phase 2
High-performance risk analytics for portfolio management
'''

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from scipy import stats

class RiskMethod(Enum):
    """Risk calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"

@dataclass
class RiskMetrics:
    """Container for all risk metrics"""
    returns: np.ndarray
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    correlation: Optional[float] = None
    volatility: float = 0.0
    downside_deviation: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: Optional[float] = None

class RiskCalculator:
    '''Advanced risk metrics calculator for portfolio management'''
    
    def __init__(self, risk_free_rate: float = 0.02, trading_days: int = 252):
        '''
        Initialize risk calculator
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
            trading_days: Number of trading days per year (default 252)
        '''
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.daily_rf = risk_free_rate / trading_days
    
    # ============== RETURN CALCULATIONS ==============
    
    @staticmethod
    def calculate_returns(prices: np.ndarray, method: str = 'simple') -> np.ndarray:
        '''
        Calculate returns from prices
        
        Args:
            prices: Array of prices
            method: 'simple' or 'log' returns
            
        Returns:
            Array of returns
        '''
        if method == 'log':
            return np.diff(np.log(prices))
        else:
            return np.diff(prices) / prices[:-1]
    
    # ============== RISK-ADJUSTED RETURNS ==============
    
    def sharpe_ratio(self, returns: np.ndarray, 
                    excess_returns: Optional[np.ndarray] = None) -> float:
        '''
        Calculate Sharpe ratio
        
        Args:
            returns: Array of returns
            excess_returns: Optional pre-calculated excess returns
            
        Returns:
            Annualized Sharpe ratio
        '''
        if excess_returns is None:
            excess_returns = returns - self.daily_rf
        
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.trading_days)
    
    def sortino_ratio(self, returns: np.ndarray,
                     target_return: Optional[float] = None) -> float:
        '''
        Calculate Sortino ratio (downside risk-adjusted return)
        
        Args:
            returns: Array of returns
            target_return: Minimum acceptable return (default: risk-free rate)
            
        Returns:
            Annualized Sortino ratio
        '''
        if target_return is None:
            target_return = self.daily_rf
        
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(downside_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(self.trading_days)
    
    def calmar_ratio(self, returns: np.ndarray, 
                    max_drawdown: Optional[float] = None) -> float:
        '''
        Calculate Calmar ratio (return over max drawdown)
        
        Args:
            returns: Array of returns
            max_drawdown: Pre-calculated max drawdown
            
        Returns:
            Calmar ratio
        '''
        if max_drawdown is None:
            _, max_drawdown, _ = self.calculate_drawdown(returns)
        
        if max_drawdown == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0
        
        annual_return = np.mean(returns) * self.trading_days
        return annual_return / abs(max_drawdown)
    
    # ============== VALUE AT RISK (VaR) ==============
    
    def calculate_var(self, returns: np.ndarray,
                     confidence_level: float = 0.95,
                     method: RiskMethod = RiskMethod.HISTORICAL) -> float:
        '''
        Calculate Value at Risk
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: Calculation method
            
        Returns:
            VaR value (negative number representing potential loss)
        '''
        if method == RiskMethod.HISTORICAL:
            return self._var_historical(returns, confidence_level)
        elif method == RiskMethod.PARAMETRIC:
            return self._var_parametric(returns, confidence_level)
        elif method == RiskMethod.MONTE_CARLO:
            return self._var_monte_carlo(returns, confidence_level)
        elif method == RiskMethod.CORNISH_FISHER:
            return self._var_cornish_fisher(returns, confidence_level)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    @staticmethod
    def _var_historical(returns: np.ndarray, confidence_level: float) -> float:
        '''Historical VaR calculation'''
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def _var_parametric(returns: np.ndarray, confidence_level: float) -> float:
        '''Parametric (variance-covariance) VaR'''
        mean = np.mean(returns)
        std = np.std(returns)
        z_score = stats.norm.ppf(1 - confidence_level)
        return mean + z_score * std
    
    def _var_monte_carlo(self, returns: np.ndarray, 
                        confidence_level: float,
                        simulations: int = 10000) -> float:
        '''Monte Carlo VaR'''
        mean = np.mean(returns)
        std = np.std(returns)
        simulated = np.random.normal(mean, std, simulations)
        return np.percentile(simulated, (1 - confidence_level) * 100)
    
    def _var_cornish_fisher(self, returns: np.ndarray, confidence_level: float) -> float:
        '''Cornish-Fisher VaR (adjusts for skewness and kurtosis)'''
        mean = np.mean(returns)
        std = np.std(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        z = stats.norm.ppf(1 - confidence_level)
        
        # Cornish-Fisher expansion
        z_cf = (z + 
                (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)
        
        return mean + z_cf * std
    
    # ============== CONDITIONAL VAR (CVaR) ==============
    
    def calculate_cvar(self, returns: np.ndarray,
                      confidence_level: float = 0.95) -> float:
        '''
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level
            
        Returns:
            CVaR value
        '''
        var = self.calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    # ============== DRAWDOWN ANALYSIS ==============
    
    @staticmethod
    def calculate_drawdown(returns: np.ndarray) -> Tuple[np.ndarray, float, int]:
        '''
        Calculate drawdown series and statistics
        
        Args:
            returns: Array of returns
            
        Returns:
            Tuple of (drawdown series, max drawdown, max duration)
        '''
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = np.min(drawdown)
        
        # Calculate maximum drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = []
        current_duration = 0
        
        for dd in is_drawdown:
            if dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    drawdown_periods.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            drawdown_periods.append(current_duration)
        
        max_duration = max(drawdown_periods) if drawdown_periods else 0
        
        return drawdown, max_drawdown, max_duration
    
    # ============== BETA & CORRELATION ==============
    
    @staticmethod
    def calculate_beta(returns: np.ndarray, 
                      market_returns: np.ndarray) -> Tuple[float, float]:
        '''
        Calculate beta and alpha relative to market
        
        Args:
            returns: Portfolio returns
            market_returns: Market benchmark returns
            
        Returns:
            Tuple of (beta, alpha)
        '''
        # Ensure same length
        min_len = min(len(returns), len(market_returns))
        returns = returns[:min_len]
        market_returns = market_returns[:min_len]
        
        # Calculate covariance and variance
        covariance = np.cov(returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 0.0, 0.0
        
        beta = covariance / market_variance
        
        # Calculate alpha (Jensen's alpha)
        alpha = np.mean(returns) - beta * np.mean(market_returns)
        
        return beta, alpha
    
    @staticmethod
    def calculate_correlation(returns1: np.ndarray, 
                            returns2: np.ndarray) -> float:
        '''
        Calculate correlation between two return series
        
        Args:
            returns1: First return series
            returns2: Second return series
            
        Returns:
            Correlation coefficient
        '''
        min_len = min(len(returns1), len(returns2))
        return np.corrcoef(returns1[:min_len], returns2[:min_len])[0, 1]
    
    # ============== VOLATILITY METRICS ==============
    
    @staticmethod
    def calculate_volatility(returns: np.ndarray, 
                           annualize: bool = True,
                           trading_days: int = 252) -> float:
        '''
        Calculate volatility (standard deviation of returns)
        
        Args:
            returns: Array of returns
            annualize: Whether to annualize the volatility
            trading_days: Number of trading days for annualization
            
        Returns:
            Volatility
        '''
        vol = np.std(returns)
        if annualize:
            vol *= np.sqrt(trading_days)
        return vol
    
    @staticmethod
    def calculate_downside_deviation(returns: np.ndarray,
                                   target_return: float = 0,
                                   annualize: bool = True,
                                   trading_days: int = 252) -> float:
        '''
        Calculate downside deviation
        
        Args:
            returns: Array of returns
            target_return: Minimum acceptable return
            annualize: Whether to annualize
            trading_days: Number of trading days
            
        Returns:
            Downside deviation
        '''
        downside_returns = returns[returns < target_return] - target_return
        
        if len(downside_returns) == 0:
            return 0.0
        
        dd = np.sqrt(np.mean(downside_returns ** 2))
        
        if annualize:
            dd *= np.sqrt(trading_days)
        
        return dd
    
    # ============== INFORMATION RATIO ==============
    
    def information_ratio(self, returns: np.ndarray,
                         benchmark_returns: np.ndarray) -> float:
        '''
        Calculate Information Ratio (active return / tracking error)
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        '''
        min_len = min(len(returns), len(benchmark_returns))
        active_returns = returns[:min_len] - benchmark_returns[:min_len]
        
        if len(active_returns) == 0 or np.std(active_returns) == 0:
            return 0.0
        
        return np.mean(active_returns) / np.std(active_returns) * np.sqrt(self.trading_days)
    
    # ============== COMPREHENSIVE ANALYSIS ==============
    
    def calculate_all_metrics(self, 
                            prices: np.ndarray,
                            benchmark_prices: Optional[np.ndarray] = None) -> RiskMetrics:
        '''
        Calculate all risk metrics
        
        Args:
            prices: Portfolio price series
            benchmark_prices: Optional benchmark price series
            
        Returns:
            RiskMetrics object with all calculations
        '''
        returns = self.calculate_returns(prices)
        
        # Basic metrics
        sharpe = self.sharpe_ratio(returns)
        sortino = self.sortino_ratio(returns)
        
        # Drawdown analysis
        _, max_dd, max_dd_duration = self.calculate_drawdown(returns)
        
        # Value at Risk
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        cvar_99 = self.calculate_cvar(returns, 0.99)
        
        # Volatility
        volatility = self.calculate_volatility(returns)
        downside_dev = self.calculate_downside_deviation(returns)
        
        # Calmar ratio
        calmar = self.calmar_ratio(returns, max_dd)
        
        # Market-relative metrics
        beta, alpha, correlation, info_ratio = None, None, None, None
        if benchmark_prices is not None:
            benchmark_returns = self.calculate_returns(benchmark_prices)
            beta, alpha = self.calculate_beta(returns, benchmark_returns)
            correlation = self.calculate_correlation(returns, benchmark_returns)
            info_ratio = self.information_ratio(returns, benchmark_returns)
        
        return RiskMetrics(
            returns=returns,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            beta=beta,
            alpha=alpha,
            correlation=correlation,
            volatility=volatility,
            downside_deviation=downside_dev,
            calmar_ratio=calmar,
            information_ratio=info_ratio
        )

def test_risk_metrics():
    '''Test the risk metrics calculations'''
    np.random.seed(42)
    
    # Generate sample price data
    n = 1000
    returns = np.random.normal(0.0005, 0.01, n)  # Daily returns
    prices = 100 * (1 + returns).cumprod()
    
    # Generate benchmark
    benchmark_returns = returns + np.random.normal(0, 0.005, n)
    benchmark_prices = 100 * (1 + benchmark_returns).cumprod()
    
    # Calculate all metrics
    calc = RiskCalculator()
    metrics = calc.calculate_all_metrics(prices, benchmark_prices)
    
    print('Risk Metrics Analysis')
    print('=' * 60)
    print(f'Sharpe Ratio:        {metrics.sharpe_ratio:.4f}')
    print(f'Sortino Ratio:       {metrics.sortino_ratio:.4f}')
    print(f'Calmar Ratio:        {metrics.calmar_ratio:.4f}')
    print(f'Information Ratio:   {metrics.information_ratio:.4f}')
    print()
    print(f'Max Drawdown:        {metrics.max_drawdown:.2%}')
    print(f'Max DD Duration:     {metrics.max_drawdown_duration} days')
    print()
    print(f'VaR (95%):          {metrics.var_95:.2%}')
    print(f'VaR (99%):          {metrics.var_99:.2%}')
    print(f'CVaR (95%):         {metrics.cvar_95:.2%}')
    print(f'CVaR (99%):         {metrics.cvar_99:.2%}')
    print()
    print(f'Volatility (Annual): {metrics.volatility:.2%}')
    print(f'Downside Deviation:  {metrics.downside_deviation:.2%}')
    print()
    if metrics.beta is not None:
        print(f'Beta:               {metrics.beta:.4f}')
        print(f'Alpha (Annual):     {metrics.alpha*252:.2%}')
        print(f'Correlation:        {metrics.correlation:.4f}')
    
    print('\n✅ All risk metrics calculated successfully!')

if __name__ == '__main__':
    test_risk_metrics()

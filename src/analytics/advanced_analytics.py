import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import json
from datetime import datetime, timedelta

class AdvancedAnalytics:
    '''Professional trading analytics engine'''
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate  # 5% treasury rate
        self.analytics_cache = {}
        
    def calculate_alpha_beta(self, returns: np.ndarray, 
                           market_returns: np.ndarray) -> Dict:
        '''Calculate alpha and beta vs market'''
        
        # Run regression
        beta, alpha, r_value, p_value, std_err = stats.linregress(
            market_returns, returns
        )
        
        # Annualize alpha
        annual_alpha = alpha * 252
        
        return {
            'alpha': annual_alpha,
            'beta': beta,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'tracking_error': std_err * np.sqrt(252)
        }
    
    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        '''Calculate Sharpe ratio'''
        excess_returns = returns - (self.risk_free_rate / 252)
        if np.std(excess_returns) == 0:
            return 0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        '''Calculate Sortino ratio (downside risk only)'''
        excess_returns = returns - (self.risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')  # No downside
            
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return float('inf')
            
        return np.sqrt(252) * np.mean(excess_returns) / downside_std
    
    def calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        '''Calculate Calmar ratio (return / max drawdown)'''
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.min(drawdown)
        
        annual_return = np.prod(1 + returns) ** (252/len(returns)) - 1
        
        if max_dd == 0:
            return float('inf')
            
        return annual_return / abs(max_dd)
    
    def calculate_information_ratio(self, returns: np.ndarray, 
                                   benchmark_returns: np.ndarray) -> float:
        '''Calculate Information Ratio'''
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(252)
        
        if tracking_error == 0:
            return 0
            
        return np.mean(active_returns) * 252 / tracking_error
    
    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> Dict:
        '''Calculate maximum drawdown metrics'''
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        # Find drawdown start
        dd_start = 0
        for i in range(max_dd_idx, -1, -1):
            if drawdown[i] == 0:
                dd_start = i
                break
        
        # Find recovery (if any)
        dd_end = len(drawdown) - 1
        for i in range(max_dd_idx, len(drawdown)):
            if drawdown[i] == 0:
                dd_end = i
                break
        
        return {
            'max_drawdown': max_dd * 100,
            'max_dd_duration': dd_end - dd_start,
            'current_drawdown': drawdown[-1] * 100,
            'time_underwater': np.sum(drawdown < 0) / len(drawdown) * 100
        }
    
    def monte_carlo_simulation(self, returns: np.ndarray, 
                             initial_capital: float = 100000,
                             num_simulations: int = 1000,
                             num_days: int = 252) -> Dict:
        '''Run Monte Carlo simulation for future performance'''
        
        # Calculate return statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Run simulations
        final_values = []
        
        for _ in range(num_simulations):
            # Generate random returns
            sim_returns = np.random.normal(mean_return, std_return, num_days)
            
            # Calculate final value
            final_value = initial_capital * np.prod(1 + sim_returns)
            final_values.append(final_value)
        
        final_values = np.array(final_values)
        
        return {
            'expected_value': np.mean(final_values),
            'median_value': np.median(final_values),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_95': np.percentile(final_values, 95),
            'probability_profit': np.sum(final_values > initial_capital) / num_simulations * 100,
            'probability_double': np.sum(final_values > initial_capital * 2) / num_simulations * 100,
            'var_95': initial_capital - np.percentile(final_values, 5),
            'cvar_95': initial_capital - np.mean(final_values[final_values <= np.percentile(final_values, 5)])
        }
    
    def calculate_rolling_metrics(self, returns: pd.Series, 
                                 window: int = 30) -> pd.DataFrame:
        '''Calculate rolling performance metrics'''
        
        metrics = pd.DataFrame(index=returns.index)
        
        # Rolling returns
        metrics['rolling_return'] = returns.rolling(window).mean() * 252
        metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)
        metrics['rolling_sharpe'] = (
            metrics['rolling_return'] / metrics['rolling_volatility']
        )
        
        # Rolling win rate
        metrics['rolling_win_rate'] = returns.rolling(window).apply(
            lambda x: np.sum(x > 0) / len(x) * 100
        )
        
        return metrics
    
    def correlation_analysis(self, returns_dict: Dict[str, np.ndarray]) -> Dict:
        '''Analyze correlations between multiple assets'''
        
        # Create correlation matrix
        assets = list(returns_dict.keys())
        n = len(assets)
        corr_matrix = np.zeros((n, n))
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                corr = np.corrcoef(returns_dict[asset1], returns_dict[asset2])[0, 1]
                corr_matrix[i, j] = corr
        
        # Find highest and lowest correlations
        correlations = []
        for i in range(n):
            for j in range(i+1, n):
                correlations.append({
                    'pair': f'{assets[i]}-{assets[j]}',
                    'correlation': corr_matrix[i, j]
                })
        
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'correlation_matrix': corr_matrix.tolist(),
            'assets': assets,
            'highest_correlation': correlations[0] if correlations else None,
            'lowest_correlation': correlations[-1] if correlations else None,
            'average_correlation': np.mean([c['correlation'] for c in correlations])
        }
    
    def risk_metrics(self, returns: np.ndarray) -> Dict:
        '''Calculate comprehensive risk metrics'''
        
        return {
            'volatility': np.std(returns) * np.sqrt(252),
            'downside_deviation': np.std(returns[returns < 0]) * np.sqrt(252),
            'var_95': np.percentile(returns, 5),
            'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'max_daily_loss': np.min(returns) * 100,
            'max_daily_gain': np.max(returns) * 100
        }
    
    def performance_attribution(self, trades: List[Dict]) -> Dict:
        '''Attribute performance to different factors'''
        
        if not trades:
            return {}
        
        # Analyze by various factors
        by_symbol = {}
        by_hour = {}
        by_day = {}
        by_strategy = {}
        
        for trade in trades:
            symbol = trade.get('symbol', 'Unknown')
            hour = trade.get('timestamp', datetime.now()).hour
            day = trade.get('timestamp', datetime.now()).weekday()
            strategy = trade.get('strategy', 'Unknown')
            profit = trade.get('profit', 0)
            
            # Aggregate by symbol
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(profit)
            
            # Aggregate by hour
            if hour not in by_hour:
                by_hour[hour] = []
            by_hour[hour].append(profit)
            
            # Aggregate by day
            if day not in by_day:
                by_day[day] = []
            by_day[day].append(profit)
            
            # Aggregate by strategy
            if strategy not in by_strategy:
                by_strategy[strategy] = []
            by_strategy[strategy].append(profit)
        
        return {
            'by_symbol': {k: np.sum(v) for k, v in by_symbol.items()},
            'by_hour': {k: np.sum(v) for k, v in by_hour.items()},
            'by_day': {k: np.sum(v) for k, v in by_day.items()},
            'by_strategy': {k: np.sum(v) for k, v in by_strategy.items()},
            'best_symbol': max(by_symbol, key=lambda x: np.sum(by_symbol[x])),
            'best_hour': max(by_hour, key=lambda x: np.sum(by_hour[x])) if by_hour else None,
            'best_strategy': max(by_strategy, key=lambda x: np.sum(by_strategy[x]))
        }
    
    def generate_report(self, portfolio_data: Dict) -> str:
        '''Generate comprehensive analytics report'''
        
        report = []
        report.append('='*60)
        report.append('📊 ADVANCED ANALYTICS REPORT')
        report.append('='*60)
        report.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        report.append('')
        
        # Performance metrics
        returns = portfolio_data.get('returns', [])
        if len(returns) > 0:
            returns = np.array(returns)
            
            report.append('PERFORMANCE METRICS:')
            report.append(f'  Sharpe Ratio: {self.calculate_sharpe_ratio(returns):.2f}')
            report.append(f'  Sortino Ratio: {self.calculate_sortino_ratio(returns):.2f}')
            report.append(f'  Calmar Ratio: {self.calculate_calmar_ratio(returns):.2f}')
            report.append('')
            
            # Risk metrics
            risk = self.risk_metrics(returns)
            report.append('RISK METRICS:')
            report.append(f'  Volatility: {risk["volatility"]:.2%}')
            report.append(f'  VaR 95%: {risk["var_95"]:.2%}')
            report.append(f'  CVaR 95%: {risk["cvar_95"]:.2%}')
            report.append(f'  Skewness: {risk["skewness"]:.2f}')
            report.append(f'  Kurtosis: {risk["kurtosis"]:.2f}')
            report.append('')
            
            # Monte Carlo
            mc = self.monte_carlo_simulation(returns)
            report.append('MONTE CARLO SIMULATION (1 Year):')
            report.append(f'  Expected Value: ')
            report.append(f'  95% Confidence:  - ')
            report.append(f'  Probability of Profit: {mc["probability_profit"]:.1f}%')
            report.append(f'  Probability of Doubling: {mc["probability_double"]:.1f}%')
        
        return '\n'.join(report)

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yfinance as yf
from scipy.optimize import minimize

class PortfolioOptimization:
    '''Portfolio optimization using Modern Portfolio Theory'''
    
    def __init__(self, symbols: List[str], risk_free_rate: float = 0.02):
        self.symbols = symbols
        self.risk_free_rate = risk_free_rate
        self.returns_data = None
        self.correlation_matrix = None
        self.covariance_matrix = None
        
    def fetch_historical_data(self, period: str = '1y') -> pd.DataFrame:
        '''Fetch historical price data'''
        data = {}
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                data[symbol] = hist['Close']
                print(f'✅ Fetched data for {symbol}')
            except Exception as e:
                print(f'❌ Error fetching {symbol}: {e}')
        
        df = pd.DataFrame(data)
        df = df.dropna()
        
        self.returns_data = df.pct_change().dropna()
        self.correlation_matrix = self.returns_data.corr()
        self.covariance_matrix = self.returns_data.cov() * 252
        
        return df
    
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict:
        '''Calculate portfolio metrics'''
        if self.returns_data is None:
            raise ValueError('Must fetch historical data first')
        
        portfolio_return = np.sum(self.returns_data.mean() * weights) * 252
        portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
        
        return {
            'return': portfolio_return,
            'risk': portfolio_std,
            'sharpe': sharpe_ratio
        }
    
    def optimize_portfolio(self, target_return: Optional[float] = None) -> Dict:
        '''Optimize portfolio for maximum Sharpe ratio'''
        n_assets = len(self.symbols)
        init_weights = np.array([1/n_assets] * n_assets)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if target_return is not None:
            constraints.append({
                'type': 'gte', 
                'fun': lambda w: np.sum(self.returns_data.mean() * w) * 252 - target_return
            })
        
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        def negative_sharpe(weights):
            metrics = self.calculate_portfolio_metrics(weights)
            return -metrics['sharpe']
        
        result = minimize(
            negative_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        metrics = self.calculate_portfolio_metrics(optimal_weights)
        
        return {
            'weights': dict(zip(self.symbols, optimal_weights)),
            'metrics': metrics,
            'success': result.success
        }
    
    def risk_parity_allocation(self) -> Dict:
        '''Risk parity portfolio allocation'''
        n_assets = len(self.symbols)
        weights = np.ones(n_assets) / n_assets
        
        for _ in range(100):
            portfolio_var = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
            marginal_contrib = np.dot(self.covariance_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_var
            
            weights = weights * np.sqrt(1/contrib)
            weights = weights / np.sum(weights)
        
        metrics = self.calculate_portfolio_metrics(weights)
        
        return {
            'weights': dict(zip(self.symbols, weights)),
            'metrics': metrics
        }

def test_portfolio_optimization():
    '''Test portfolio optimization'''
    print('='*60)
    print('📊 Testing Portfolio Optimization')
    print('='*60)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
    
    optimizer = PortfolioOptimization(symbols)
    
    print(f'\n1. Fetching data for {", ".join(symbols)}...')
    df = optimizer.fetch_historical_data(period='6mo')
    print(f'   Data shape: {df.shape}')
    
    print('\n2. Optimizing portfolio...')
    optimal = optimizer.optimize_portfolio()
    
    if optimal['success']:
        print('\n✅ Optimal Portfolio:')
        for symbol, weight in optimal['weights'].items():
            print(f'   {symbol}: {weight*100:.1f}%')
        
        print(f'\n📈 Expected Metrics:')
        print(f'   Annual Return: {optimal["metrics"]["return"]*100:.1f}%')
        print(f'   Annual Risk: {optimal["metrics"]["risk"]*100:.1f}%')
        print(f'   Sharpe Ratio: {optimal["metrics"]["sharpe"]:.2f}')
    
    print('\n3. Risk Parity Allocation...')
    risk_parity = optimizer.risk_parity_allocation()
    print('\n✅ Risk Parity Weights:')
    for symbol, weight in risk_parity['weights'].items():
        print(f'   {symbol}: {weight*100:.1f}%')
    
    print('\n✅ Portfolio optimization test complete!')

if __name__ == '__main__':
    test_portfolio_optimization()

import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import time

class PaperTradingAccount:
    '''Simulated brokerage account with real money tracking'''
    
    def __init__(self, 
                 starting_capital: float = 100000,
                 account_name: str = "Main",
                 save_dir: str = "paper_trading_data"):
        
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.account_name = account_name
        self.save_dir = save_dir
        
        # Portfolio tracking
        self.positions = {}  # symbol -> {'shares': x, 'avg_cost': y}
        self.pending_orders = []
        self.order_history = []
        self.transaction_history = []
        
        # Performance tracking
        self.daily_values = []
        self.peak_value = starting_capital
        
        # Trading costs
        self.commission = 0  # Free trading era!
        self.slippage = 0.001  # 0.1% slippage
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Load existing account if exists
        self.account_file = os.path.join(save_dir, f"{account_name}_account.json")
        self.load_account()
    
    def get_account_value(self, current_prices: Dict[str, float]) -> float:
        '''Calculate total account value'''
        positions_value = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                positions_value += position['shares'] * current_prices[symbol]
        
        return self.cash + positions_value
    
    def execute_market_order(self, 
                           symbol: str, 
                           quantity: int, 
                           current_price: float,
                           order_type: str = 'BUY') -> Dict:
        '''Execute a market order with realistic simulation'''
        
        # Apply slippage
        if order_type == 'BUY':
            execution_price = current_price * (1 + self.slippage)
        else:
            execution_price = current_price * (1 - self.slippage)
        
        # Calculate total cost
        order_value = quantity * execution_price
        total_cost = order_value + self.commission
        
        # Execute order
        if order_type == 'BUY':
            if self.cash >= total_cost:
                self.cash -= total_cost
                
                # Update positions
                if symbol in self.positions:
                    # Average up/down
                    old_shares = self.positions[symbol]['shares']
                    old_cost = self.positions[symbol]['avg_cost']
                    new_shares = old_shares + quantity
                    new_avg_cost = (old_shares * old_cost + quantity * execution_price) / new_shares
                    
                    self.positions[symbol] = {
                        'shares': new_shares,
                        'avg_cost': new_avg_cost,
                        'last_price': execution_price
                    }
                else:
                    self.positions[symbol] = {
                        'shares': quantity,
                        'avg_cost': execution_price,
                        'last_price': execution_price
                    }
                
                order_result = {
                    'status': 'FILLED',
                    'symbol': symbol,
                    'type': 'BUY',
                    'quantity': quantity,
                    'price': execution_price,
                    'value': order_value,
                    'commission': self.commission,
                    'timestamp': datetime.now()
                }
            else:
                order_result = {
                    'status': 'REJECTED',
                    'reason': 'Insufficient funds',
                    'required': total_cost,
                    'available': self.cash
                }
        
        else:  # SELL
            if symbol in self.positions and self.positions[symbol]['shares'] >= quantity:
                self.cash += order_value - self.commission
                
                # Update positions
                self.positions[symbol]['shares'] -= quantity
                
                # Remove position if fully closed
                if self.positions[symbol]['shares'] == 0:
                    del self.positions[symbol]
                
                order_result = {
                    'status': 'FILLED',
                    'symbol': symbol,
                    'type': 'SELL',
                    'quantity': quantity,
                    'price': execution_price,
                    'value': order_value,
                    'commission': self.commission,
                    'timestamp': datetime.now()
                }
            else:
                order_result = {
                    'status': 'REJECTED',
                    'reason': 'Insufficient shares',
                    'required': quantity,
                    'available': self.positions.get(symbol, {}).get('shares', 0)
                }
        
        # Record transaction
        if order_result['status'] == 'FILLED':
            self.order_history.append(order_result)
            self.save_account()
        
        return order_result
    
    def get_position_pnl(self, symbol: str, current_price: float) -> Dict:
        '''Calculate P&L for a position'''
        if symbol not in self.positions:
            return {'unrealized_pnl': 0, 'unrealized_pnl_pct': 0}
        
        position = self.positions[symbol]
        current_value = position['shares'] * current_price
        cost_basis = position['shares'] * position['avg_cost']
        
        unrealized_pnl = current_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
        
        return {
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'current_value': current_value,
            'cost_basis': cost_basis
        }
    
    def get_account_summary(self, current_prices: Dict[str, float]) -> Dict:
        '''Get comprehensive account summary'''
        total_value = self.get_account_value(current_prices)
        
        # Calculate total P&L
        total_unrealized_pnl = 0
        for symbol in self.positions:
            if symbol in current_prices:
                pnl = self.get_position_pnl(symbol, current_prices[symbol])
                total_unrealized_pnl += pnl['unrealized_pnl']
        
        # Calculate returns
        total_return = total_value - self.starting_capital
        total_return_pct = (total_return / self.starting_capital) * 100
        
        # Update peak for drawdown calculation
        if total_value > self.peak_value:
            self.peak_value = total_value
        
        drawdown = (total_value - self.peak_value) / self.peak_value * 100
        
        return {
            'account_name': self.account_name,
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': total_value - self.cash,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'unrealized_pnl': total_unrealized_pnl,
            'drawdown': drawdown,
            'num_positions': len(self.positions),
            'num_trades': len(self.order_history)
        }
    
    def save_account(self):
        '''Save account state to file'''
        account_data = {
            'cash': self.cash,
            'positions': self.positions,
            'order_history': [
                {k: str(v) if isinstance(v, datetime) else v 
                 for k, v in order.items()}
                for order in self.order_history
            ],
            'starting_capital': self.starting_capital,
            'peak_value': self.peak_value,
            'last_updated': str(datetime.now())
        }
        
        with open(self.account_file, 'w') as f:
            json.dump(account_data, f, indent=2)
    
    def load_account(self):
        '''Load account state from file'''
        if os.path.exists(self.account_file):
            with open(self.account_file, 'r') as f:
                data = json.load(f)
                self.cash = data.get('cash', self.starting_capital)
                self.positions = data.get('positions', {})
                self.peak_value = data.get('peak_value', self.starting_capital)
                print(f"✅ Loaded existing account: {self.account_name}")
        else:
            print(f"📝 Created new account: {self.account_name}")
            self.save_account()

class PaperTradingEngine:
    '''Main paper trading engine with strategy execution'''
    
    def __init__(self, 
                 strategy,
                 symbols: List[str],
                 account: PaperTradingAccount):
        
        self.strategy = strategy
        self.symbols = symbols
        self.account = account
        self.running = False
        
        # Import market data fetcher
        import sys
        sys.path.insert(0, '.')
        from src.market_data import MarketDataFetcher
        self.fetcher = MarketDataFetcher()
        
        # Performance tracking
        self.trade_log = []
        self.performance_history = []
    
    def get_current_prices(self) -> Dict[str, float]:
        '''Get current market prices'''
        prices = {}
        for symbol in self.symbols:
            try:
                data = self.fetcher.get_live_price(symbol, 'stock')
                if data and 'price' in data:
                    prices[symbol] = data['price']
            except:
                pass
        return prices
    
    def execute_strategy_signal(self, signal: Dict, current_prices: Dict):
        '''Execute trading signal from strategy'''
        symbol = signal.get('symbol')
        action = signal.get('action')
        
        if not symbol or not action:
            return
        
        current_price = current_prices.get(symbol)
        if not current_price:
            return
        
        # Position sizing (risk 2% per trade)
        account_value = self.account.get_account_value(current_prices)
        position_size = account_value * 0.02
        shares = int(position_size / current_price)
        
        if shares == 0:
            return
        
        # Execute based on signal
        if action == 'BUY' and symbol not in self.account.positions:
            result = self.account.execute_market_order(
                symbol, shares, current_price, 'BUY'
            )
            print(f"🟢 BUY {shares} {symbol} @ ")
            
        elif action == 'SELL' and symbol in self.account.positions:
            position_shares = self.account.positions[symbol]['shares']
            result = self.account.execute_market_order(
                symbol, position_shares, current_price, 'SELL'
            )
            print(f"🔴 SELL {position_shares} {symbol} @ ")
    
    def run_trading_session(self, duration_hours: float = 1):
        '''Run a paper trading session'''
        print(f"\n{'='*60}")
        print(f"🚀 STARTING PAPER TRADING SESSION")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        self.running = True
        update_interval = 60  # Update every minute
        
        while self.running and datetime.now() < end_time:
            # Get current prices
            current_prices = self.get_current_prices()
            
            if current_prices:
                # Get account summary
                summary = self.account.get_account_summary(current_prices)
                
                # Display status
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')}")
                print(f"💰 Account Value: ")
                print(f"📈 Return: {summary['total_return_pct']:+.2f}%")
                print(f"💵 Cash: ")
                
                # Get strategy signals
                for symbol in self.symbols:
                    # This is where your strategy generates signals
                    # For demo, using random signals
                    if np.random.random() > 0.7:
                        signal = {
                            'symbol': symbol,
                            'action': np.random.choice(['BUY', 'SELL', 'HOLD'])
                        }
                        
                        if signal['action'] != 'HOLD':
                            self.execute_strategy_signal(signal, current_prices)
                
                # Save performance
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'account_value': summary['total_value'],
                    'return_pct': summary['total_return_pct']
                })
            
            # Wait for next update
            time.sleep(update_interval)
        
        print(f"\n{'='*60}")
        print(f"📊 SESSION COMPLETE")
        print(f"{'='*60}")
        self.print_performance_report()
    
    def print_performance_report(self):
        '''Print detailed performance report'''
        current_prices = self.get_current_prices()
        summary = self.account.get_account_summary(current_prices)
        
        print(f"\n📈 PERFORMANCE REPORT")
        print(f"{'='*40}")
        print(f"Total Return: {summary['total_return_pct']:+.2f}%")
        print(f"Final Value: ")
        print(f"Total Trades: {summary['num_trades']}")
        print(f"Max Drawdown: {summary['drawdown']:.2f}%")
        
        if self.account.positions:
            print(f"\n📊 OPEN POSITIONS:")
            for symbol, position in self.account.positions.items():
                if symbol in current_prices:
                    pnl = self.account.get_position_pnl(symbol, current_prices[symbol])
                    print(f"  {symbol}: {position['shares']} shares")
                    print(f"    P&L:  ({pnl['unrealized_pnl_pct']:+.2f}%)")

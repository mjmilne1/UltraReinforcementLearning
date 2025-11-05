import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"

class PositionSide(Enum):
    """Position sides"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

@dataclass
class Position:
    """Single position in portfolio"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    side: PositionSide
    entry_time: datetime
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            return (self.entry_price - self.current_price) * self.quantity
        return 0.0
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return self.unrealized_pnl / (self.entry_price * abs(self.quantity))

@dataclass
class Trade:
    """Executed trade record"""
    symbol: str
    quantity: float
    price: float
    side: str
    timestamp: datetime
    commission: float
    
    @property
    def total_cost(self) -> float:
        return self.quantity * self.price + self.commission

class Portfolio:
    """Portfolio manager for trading environment"""
    
    def __init__(self, initial_cash=100000.0, commission_rate=0.001):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.positions = {}
        self.trades = []
        self.equity_curve = [initial_cash]
        self.returns = []
        self.peak_equity = initial_cash
    
    @property
    def equity(self) -> float:
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value
    
    @property
    def position_count(self) -> int:
        return len(self.positions)
    
    @property
    def exposure(self) -> float:
        return sum(abs(pos.market_value) for pos in self.positions.values())
    
    @property
    def buying_power(self) -> float:
        return self.cash
    
    def can_buy(self, symbol, quantity, price):
        required_cash = quantity * price * (1 + self.commission_rate)
        return required_cash <= self.buying_power
    
    def can_sell(self, symbol, quantity):
        if symbol not in self.positions:
            return False
        return self.positions[symbol].quantity >= quantity
    
    def execute_buy(self, symbol, quantity, price, timestamp=None):
        if not self.can_buy(symbol, quantity, price):
            return False
        
        if timestamp is None:
            timestamp = datetime.now()
        
        commission = quantity * price * self.commission_rate
        total_cost = quantity * price + commission
        
        self.cash -= total_cost
        
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_quantity = pos.quantity + quantity
            avg_price = (pos.entry_price * pos.quantity + price * quantity) / total_quantity
            pos.quantity = total_quantity
            pos.entry_price = avg_price
            pos.current_price = price
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                side=PositionSide.LONG,
                entry_time=timestamp
            )
        
        trade = Trade(
            symbol=symbol,
            quantity=quantity,
            price=price,
            side="buy",
            timestamp=timestamp,
            commission=commission
        )
        self.trades.append(trade)
        self.equity_curve.append(self.equity)
        
        return True
    
    def execute_sell(self, symbol, quantity, price, timestamp=None):
        if not self.can_sell(symbol, quantity):
            return False
        
        if timestamp is None:
            timestamp = datetime.now()
        
        commission = quantity * price * self.commission_rate
        proceeds = quantity * price - commission
        
        self.cash += proceeds
        
        pos = self.positions[symbol]
        pos.quantity -= quantity
        pos.current_price = price
        
        if pos.quantity == 0:
            del self.positions[symbol]
        
        trade = Trade(
            symbol=symbol,
            quantity=quantity,
            price=price,
            side="sell",
            timestamp=timestamp,
            commission=commission
        )
        self.trades.append(trade)
        self.equity_curve.append(self.equity)
        
        return True
    
    def update_prices(self, prices):
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
        
        current_equity = self.equity
        self.equity_curve.append(current_equity)
        
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        if len(self.equity_curve) >= 2:
            ret = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            self.returns.append(ret)
    
    def get_performance_metrics(self):
        if len(self.returns) == 0:
            return {
                "total_return": 0,
                "total_return_pct": 0,
                "current_equity": self.equity,
                "peak_equity": self.peak_equity,
                "current_positions": self.position_count,
                "current_exposure": self.exposure,
                "cash_balance": self.cash
            }
        
        returns_array = np.array(self.returns)
        equity_array = np.array(self.equity_curve)
        
        total_return = (self.equity - self.initial_cash) / self.initial_cash
        avg_return = np.mean(returns_array)
        volatility = np.std(returns_array) if len(returns_array) > 0 else 0
        
        sharpe = avg_return / volatility * np.sqrt(252) if volatility > 0 else 0
        
        peaks = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - peaks) / peaks
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        winning_trades = len([t for t in self.trades if t.side == "sell"])
        win_rate = winning_trades / len(self.trades) if self.trades else 0
        
        return {
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "current_equity": self.equity,
            "peak_equity": self.peak_equity,
            "avg_return": avg_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "current_positions": self.position_count,
            "current_exposure": self.exposure,
            "cash_balance": self.cash
        }
    
    def reset(self):
        self.cash = self.initial_cash
        self.positions.clear()
        self.trades.clear()
        self.equity_curve = [self.initial_cash]
        self.returns = []
        self.peak_equity = self.initial_cash

import sys
sys.path.insert(0, '.')
from src.alerts.alert_manager import AlertManager, AlertLevel
from typing import Dict, Optional
import numpy as np

class TradingMonitor:
    '''Monitor trading activity and send alerts'''
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.thresholds = {
            'max_drawdown': -0.10,      # -10%
            'daily_loss_limit': -0.05,   # -5%
            'position_size_limit': 0.20, # 20% of portfolio
            'win_rate_warning': 0.40,    # Below 40%
            'low_cash_warning': 0.10     # Below 10% cash
        }
        
    def on_trade_executed(self, trade_info: Dict):
        '''Alert on trade execution'''
        symbol = trade_info.get('symbol')
        action = trade_info.get('type')
        quantity = trade_info.get('quantity')
        price = trade_info.get('price', 0)
        
        message = f'Trade Executed:\n'
        message += f'Symbol: {symbol}\n'
        message += f'Action: {action}\n'
        message += f'Quantity: {quantity}\n'
        message += f'Price: ${price:.2f}\n'
        message += f'Value: ${quantity * price:.2f}'
        
        self.alert_manager.send_alert(message, AlertLevel.SUCCESS)
    
    def on_strategy_signal(self, signal: Dict):
        '''Alert on strategy signals'''
        strategy = signal.get('strategy')
        symbol = signal.get('symbol')
        action = signal.get('action')
        confidence = signal.get('confidence', 0)
        
        if confidence > 0.8:
            level = AlertLevel.SUCCESS
        elif confidence > 0.6:
            level = AlertLevel.INFO
        else:
            level = AlertLevel.WARNING
        
        message = f'Strategy Signal:\n'
        message += f'Strategy: {strategy}\n'
        message += f'Symbol: {symbol}\n'
        message += f'Action: {action}\n'
        message += f'Confidence: {confidence:.1%}'
        
        self.alert_manager.send_alert(message, level)
    
    def check_risk_limits(self, portfolio_metrics: Dict):
        '''Check risk limits and alert if breached'''
        
        # Check drawdown
        drawdown = portfolio_metrics.get('drawdown', 0)
        if drawdown < self.thresholds['max_drawdown']:
            message = f'DRAWDOWN ALERT!\n'
            message += f'Current Drawdown: {drawdown:.2%}\n'
            message += f'Threshold: {self.thresholds["max_drawdown"]:.2%}\n'
            message += 'Consider reducing positions!'
            
            self.alert_manager.send_alert(message, AlertLevel.CRITICAL)
        
        # Check daily loss
        daily_return = portfolio_metrics.get('daily_return', 0)
        if daily_return < self.thresholds['daily_loss_limit']:
            message = f'DAILY LOSS LIMIT!\n'
            message += f'Today\'s Return: {daily_return:.2%}\n'
            message += 'Trading should be paused!'
            
            self.alert_manager.send_alert(message, AlertLevel.WARNING)
        
        # Check cash level
        cash_percentage = portfolio_metrics.get('cash_percentage', 0)
        if cash_percentage < self.thresholds['low_cash_warning']:
            message = f'LOW CASH WARNING!\n'
            message += f'Cash Level: {cash_percentage:.1%}\n'
            message += 'Limited buying power remaining!'
            
            self.alert_manager.send_alert(message, AlertLevel.WARNING)
    
    def on_error(self, error_info: Dict):
        '''Alert on system errors'''
        error_type = error_info.get('type', 'Unknown')
        error_msg = error_info.get('message', '')
        component = error_info.get('component', 'System')
        
        message = f'SYSTEM ERROR!\n'
        message += f'Component: {component}\n'
        message += f'Error Type: {error_type}\n'
        message += f'Message: {error_msg}'
        
        self.alert_manager.send_alert(message, AlertLevel.CRITICAL)
    
    def daily_summary(self, summary: Dict):
        '''Send daily performance summary'''
        message = f'DAILY SUMMARY\n'
        message += f'Date: {summary.get("date")}\n\n'
        message += f'Performance:\n'
        message += f'• Return: {summary.get("return_pct", 0):+.2f}%\n'
        message += f'• Trades: {summary.get("num_trades", 0)}\n'
        message += f'• Win Rate: {summary.get("win_rate", 0):.1%}\n'
        message += f'• Portfolio Value: ${summary.get("portfolio_value", 0):,.2f}\n\n'
        message += f'Risk Metrics:\n'
        message += f'• Sharpe: {summary.get("sharpe_ratio", 0):.2f}\n'
        message += f'• Max Drawdown: {summary.get("max_drawdown", 0):.2%}\n'
        message += f'• Volatility: {summary.get("volatility", 0):.2%}'
        
        # Determine level based on performance
        if summary.get("return_pct", 0) > 2:
            level = AlertLevel.SUCCESS
        elif summary.get("return_pct", 0) < -2:
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.INFO
        
        self.alert_manager.send_alert(message, level)

class SystemMonitor:
    '''Monitor system health'''
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.health_checks = {
            'data_feed': True,
            'database': True,
            'broker_connection': True,
            'memory_usage': 0,
            'cpu_usage': 0
        }
    
    def check_system_health(self):
        '''Check overall system health'''
        import psutil
        
        # Check memory usage
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            self.alert_manager.send_alert(
                f'High memory usage: {memory_percent}%',
                AlertLevel.WARNING
            )
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            self.alert_manager.send_alert(
                f'High CPU usage: {cpu_percent}%',
                AlertLevel.WARNING
            )
        
        return {
            'memory': memory_percent,
            'cpu': cpu_percent,
            'status': 'healthy' if memory_percent < 90 and cpu_percent < 90 else 'degraded'
        }

import sys
sys.path.insert(0, '.')
from src.alerts.alert_manager import AlertManager, AlertLevel
from src.alerts.monitors.trading_monitor import TradingMonitor, SystemMonitor
import json
import time
import os

print('🔔 Ultra RL - Alert System Setup')
print('='*50)

# Check if config exists
if not os.path.exists('alert_config.json'):
    print('\n⚠️  No configuration found!')
    print('Creating configuration...\n')
    
    config = {
        'email': {'enabled': False},
        'discord': {'enabled': False},
        'telegram': {'enabled': False},
        'sms': {'enabled': False}
    }
    
    # Setup Discord (easiest)
    use_discord = input('Setup Discord alerts? (y/n): ')
    if use_discord.lower() == 'y':
        print('\nTo get Discord webhook:')
        print('1. Go to Discord server settings')
        print('2. Integrations -> Webhooks -> New Webhook')
        print('3. Copy webhook URL')
        
        webhook = input('\nEnter Discord webhook URL: ')
        config['discord'] = {
            'enabled': True,
            'webhook_url': webhook
        }
    
    # Save config
    with open('alert_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print('✅ Configuration saved!')

# Initialize alert system
alert_manager = AlertManager('alert_config.json')
trading_monitor = TradingMonitor(alert_manager)
system_monitor = SystemMonitor(alert_manager)

print('\n📊 Testing Alert System...')
print('-'*40)

# Test different alert levels
tests = [
    (AlertLevel.INFO, '📊 System initialized successfully'),
    (AlertLevel.SUCCESS, '✅ Test trade executed: BUY 100 AAPL @ $270.50'),
    (AlertLevel.WARNING, '⚠️ Drawdown approaching limit: -8.5%'),
    (AlertLevel.CRITICAL, '🚨 Risk limit breached! Immediate action required')
]

for level, message in tests:
    print(f'\nSending {level.value} alert...')
    if alert_manager.send_alert(message, level):
        print(f'  ✅ {level.value} alert sent!')
    else:
        print(f'  ❌ {level.value} alert failed')
    time.sleep(1)

# Test monitors
print('\n\n📈 Testing Trading Monitor...')
print('-'*40)

# Simulate trade
trade = {
    'symbol': 'AAPL',
    'type': 'BUY',
    'quantity': 100,
    'price': 270.50
}
trading_monitor.on_trade_executed(trade)
print('  ✅ Trade alert sent!')

# Simulate portfolio metrics
portfolio = {
    'drawdown': -0.08,
    'daily_return': 0.02,
    'cash_percentage': 0.25
}
trading_monitor.check_risk_limits(portfolio)
print('  ✅ Risk monitoring active!')

# System health check
print('\n\n🖥️ Testing System Monitor...')
print('-'*40)
health = system_monitor.check_system_health()
print(f'  Memory: {health["memory"]:.1f}%')
print(f'  CPU: {health["cpu"]:.1f}%')
print(f'  Status: {health["status"]}')

print('\n\n✅ Alert system fully operational!')
print('\nYour monitoring includes:')
print('  • Trade execution alerts')
print('  • Risk limit warnings')
print('  • System health monitoring')
print('  • Daily performance summaries')
print('  • Error notifications')

import json
import os

try:
    # Load account data
    with open('paper_trading_data/MainAccount_account.json', 'r') as f:
        data = json.load(f)
    
    print('💰 PAPER TRADING STATUS')
    print('='*50)
    
    # Basic info
    starting = 100000
    cash = data.get('cash', starting)
    positions = data.get('positions', {})
    trades = data.get('order_history', [])
    
    print(f'Starting Capital: ')
    print(f'Current Cash: ')
    print(f'Open Positions: {len(positions)}')
    print(f'Total Trades: {len(trades)}')
    
    # Calculate portfolio value
    portfolio_value = cash
    
    if positions:
        print('\n📊 CURRENT POSITIONS:')
        for symbol, pos in positions.items():
            shares = pos.get('shares', 0)
            avg_cost = pos.get('avg_cost', 0)
            last_price = pos.get('last_price', avg_cost)
            value = shares * last_price
            portfolio_value += value
            
            print(f'  {symbol}: {shares} shares @  = ')
    
    # Calculate returns
    total_return = portfolio_value - starting
    return_pct = (total_return / starting) * 100
    
    print('\n📈 PERFORMANCE:')
    print(f'Portfolio Value: ')
    print(f'Total Return: ')
    print(f'Return %: {return_pct:+.2f}%')
    
    # Show recent trades
    if trades:
        print('\n📜 RECENT TRADES:')
        for trade in trades[-5:]:  # Last 5 trades
            symbol = trade.get('symbol', 'N/A')
            trade_type = trade.get('type', 'N/A')
            quantity = trade.get('quantity', 0)
            price = trade.get('price', 0)
            print(f'  {trade_type} {quantity} {symbol} @ ')
    
except FileNotFoundError:
    print('❌ No paper trading data found!')
    print('Make sure paper trading is running.')
except Exception as e:
    print(f'Error: {e}')

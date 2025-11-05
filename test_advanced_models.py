import sys
sys.path.insert(0, '.')
import numpy as np
import torch

print('='*60)
print('Testing Advanced ML Models')
print('='*60)

# Test Transformer
print('\n1. TRANSFORMER MODEL')
print('-'*40)
try:
    from src.models.transformers.market_transformer import TransformerTrader
    
    trader = TransformerTrader(input_dim=5, seq_len=20)
    market_data = np.random.randn(30, 5)
    prediction = trader.predict(market_data)
    
    if prediction:
        actions = ['HOLD', 'BUY', 'SELL']
        print('   Predicted Action:', actions[prediction['action']])
        print('   Predicted Volatility: {:.2%}'.format(prediction['predicted_volatility']))
        print('   ✅ Transformer working!')
except Exception as e:
    print('   Error:', str(e))

# Test LSTM
print('\n2. LSTM WITH ATTENTION')
print('-'*40)
try:
    from src.models.lstm.lstm_attention import LSTMPredictor
    
    predictor = LSTMPredictor(input_dim=5)
    sequence = np.random.randn(30, 5)
    prediction = predictor.predict_next(sequence)
    
    print('   Action:', prediction['action'])
    print('   Confidence: {:.2%}'.format(prediction['confidence']))
    print('   Up Probability: {:.2%}'.format(prediction['up_prob']))
    print('   ✅ LSTM working!')
except Exception as e:
    print('   Error:', str(e))

# Test GNN
print('\n3. GRAPH NEURAL NETWORK')
print('-'*40)
try:
    from src.models.gnn.graph_network import CorrelationTrader
    
    assets = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    trader = CorrelationTrader(assets)
    
    features = np.random.randn(4, 10)
    returns_data = {
        'AAPL': np.random.randn(100),
        'MSFT': np.random.randn(100),
        'GOOGL': np.random.randn(100),
        'NVDA': np.random.randn(100)
    }
    
    allocation = trader.get_portfolio_allocation(features, returns_data)
    
    if allocation:
        print('   Portfolio Allocation:')
        for asset, data in allocation.items():
            weight = data['weight'] * 100
            action = data['action']
            print('     {}: {:.1f}% - {}'.format(asset, weight, action))
        print('   ✅ GNN working!')
except Exception as e:
    print('   Error:', str(e))

print('\n' + '='*60)
print('Advanced ML Models Test Complete!')
print('='*60)

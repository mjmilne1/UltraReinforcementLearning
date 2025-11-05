import numpy as np
from typing import List, Dict, Any
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.strategies.ml_agents.ppo_agent import PPOAgent
from src.strategies.ml_agents.a2c_agent import A2CAgent
from src.agents.dqn_agent_simple import DQNAgent
from src.strategies.classical.traditional_strategies import (
    MeanReversionStrategy, 
    MomentumStrategy, 
    PairsTradingStrategy
)

class EnsembleStrategy:
    '''Combines multiple strategies using voting or weighted average'''
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize all strategies
        self.strategies = {
            'dqn': DQNAgent(state_size, action_size),
            'ppo': PPOAgent(state_size, action_size),
            'a2c': A2CAgent(state_size, action_size),
            'mean_reversion': MeanReversionStrategy(),
            'momentum': MomentumStrategy()
        }
        
        # Strategy weights (can be optimized)
        self.weights = {
            'dqn': 0.25,
            'ppo': 0.25,
            'a2c': 0.20,
            'mean_reversion': 0.15,
            'momentum': 0.15
        }
        
        # Performance tracking
        self.strategy_performance = {name: [] for name in self.strategies.keys()}
        
    def get_ml_action(self, state: np.ndarray, strategy_name: str) -> int:
        '''Get action from ML agent'''
        if strategy_name == 'dqn':
            return self.strategies[strategy_name].act(state, training=False)
        elif strategy_name in ['ppo', 'a2c']:
            return self.strategies[strategy_name].act(state, training=False)
        return 0
    
    def get_classical_action(self, prices: np.ndarray, strategy_name: str) -> int:
        '''Convert classical strategy signals to actions'''
        if strategy_name == 'mean_reversion':
            signal = self.strategies[strategy_name].generate_signal(prices)
        elif strategy_name == 'momentum':
            signal = self.strategies[strategy_name].generate_signal(prices)
        else:
            signal = 0
        
        # Convert signal to action (0=hold, 1=buy, 2=sell)
        if signal > 0:
            return 1  # Buy
        elif signal < 0:
            return 2  # Sell
        else:
            return 0  # Hold
    
    def vote(self, state: np.ndarray, prices: np.ndarray) -> int:
        '''Get ensemble action using weighted voting'''
        votes = np.zeros(self.action_size)
        
        # Collect votes from each strategy
        for name, strategy in self.strategies.items():
            if name in ['dqn', 'ppo', 'a2c']:
                action = self.get_ml_action(state, name)
            else:
                action = self.get_classical_action(prices, name)
            
            # Add weighted vote
            votes[action] += self.weights[name]
        
        # Return action with highest votes
        return np.argmax(votes)
    
    def update_weights(self, strategy_returns: Dict[str, float]):
        '''Update strategy weights based on performance'''
        # Softmax on returns for new weights
        returns = np.array([strategy_returns.get(name, 0) for name in self.strategies.keys()])
        exp_returns = np.exp(returns)
        new_weights = exp_returns / np.sum(exp_returns)
        
        # Update weights with momentum
        momentum = 0.9
        for i, name in enumerate(self.strategies.keys()):
            self.weights[name] = momentum * self.weights[name] + (1 - momentum) * new_weights[i]
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def get_confidence(self, state: np.ndarray, prices: np.ndarray) -> float:
        '''Calculate confidence in the ensemble decision'''
        votes = np.zeros(self.action_size)
        
        for name, strategy in self.strategies.items():
            if name in ['dqn', 'ppo', 'a2c']:
                action = self.get_ml_action(state, name)
            else:
                action = self.get_classical_action(prices, name)
            
            votes[action] += self.weights[name]
        
        # Confidence is the percentage of agreement
        max_vote = np.max(votes)
        confidence = max_vote / sum(self.weights.values())
        
        return confidence
    
    def get_strategy_signals(self, state: np.ndarray, prices: np.ndarray) -> Dict:
        '''Get all individual strategy signals for analysis'''
        signals = {}
        
        for name in self.strategies.keys():
            if name in ['dqn', 'ppo', 'a2c']:
                action = self.get_ml_action(state, name)
            else:
                action = self.get_classical_action(prices, name)
            
            signals[name] = ['HOLD', 'BUY', 'SELL'][action]
        
        return signals

def test_ensemble():
    '''Test ensemble strategy'''
    print('='*60)
    print('?? Testing Ensemble Strategy')
    print('='*60)
    
    # Create dummy data
    state = np.random.randn(40)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    # Initialize ensemble
    ensemble = EnsembleStrategy(state_size=40, action_size=3)
    
    # Get ensemble decision
    action = ensemble.vote(state, prices)
    confidence = ensemble.get_confidence(state, prices)
    signals = ensemble.get_strategy_signals(state, prices)
    
    print('\nIndividual Strategy Signals:')
    for name, signal in signals.items():
        print(f'  {name:15s}: {signal}')
    
    print(f'\nEnsemble Decision: {["HOLD", "BUY", "SELL"][action]}')
    print(f'Confidence: {confidence*100:.1f}%')
    
    print('\nStrategy Weights:')
    for name, weight in ensemble.weights.items():
        print(f'  {name:15s}: {weight*100:.1f}%')
    
    print('\n? Ensemble strategy test complete!')

if __name__ == '__main__':
    test_ensemble()


'''Deep Q-Network Model
Neural network architecture for Q-learning
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

class DQN(nn.Module):
    '''Deep Q-Network for trading decisions'''
    
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 hidden_layers: List[int] = [512, 256, 128]):
        '''
        Initialize DQN
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_layers: List of hidden layer sizes
        '''
        super(DQN, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build network layers
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        '''Forward pass'''
        return self.network(state)
    
    def _initialize_weights(self):
        '''Initialize network weights using He initialization'''
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0)

class DuelingDQN(nn.Module):
    '''Dueling DQN architecture for improved value estimation'''
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_layers: List[int] = [512, 256]):
        '''Initialize Dueling DQN'''
        super(DuelingDQN, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        self._initialize_weights()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        '''Forward pass with dueling architecture'''
        features = self.features(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage (Wang et al., 2016)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def _initialize_weights(self):
        '''Initialize weights'''
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0)

class NoisyLinear(nn.Module):
    '''Noisy linear layer for exploration'''
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        '''Initialize noisy linear layer'''
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # Factorized Gaussian noise
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass with noise'''
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        '''Initialize parameters'''
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        '''Reset noise'''
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        '''Generate scaled noise'''
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

class RainbowDQN(nn.Module):
    '''Rainbow DQN combining multiple improvements'''
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 num_atoms: int = 51,
                 v_min: float = -10.0,
                 v_max: float = 10.0):
        '''Initialize Rainbow DQN'''
        super(RainbowDQN, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Distributional RL support
        self.support = torch.linspace(v_min, v_max, num_atoms)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Network with noisy layers
        self.features = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            NoisyLinear(512, 512),
            nn.ReLU()
        )
        
        # Dueling streams with distributional outputs
        self.value_stream = nn.Sequential(
            NoisyLinear(512, 256),
            nn.ReLU(),
            NoisyLinear(256, num_atoms)
        )
        
        self.advantage_stream = nn.Sequential(
            NoisyLinear(512, 256),
            nn.ReLU(),
            NoisyLinear(256, action_size * num_atoms)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        '''Forward pass returning action values'''
        batch_size = state.size(0)
        
        features = self.features(state)
        
        value = self.value_stream(features).view(batch_size, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(batch_size, self.action_size, self.num_atoms)
        
        # Combine value and advantage
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probabilities
        q_dist = F.softmax(q_atoms, dim=2)
        
        # Calculate Q-values as expected values
        q_values = (q_dist * self.support.view(1, 1, -1)).sum(dim=2)
        
        return q_values
    
    def reset_noise(self):
        '''Reset noise in all noisy layers'''
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

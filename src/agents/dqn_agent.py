'''Deep Q-Network Agent for Trading
Complete implementation with training and trading logic
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List, Dict
import random
from collections import deque

from .models.dqn_model import DQN, DuelingDQN, RainbowDQN
from .memory.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class TradingAction:
    '''Trading action definitions'''
    HOLD = 0
    BUY = 1
    SELL = 2
    
    @staticmethod
    def to_string(action: int) -> str:
        actions = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        return actions.get(action, 'UNKNOWN')

class DQNAgent:
    '''DQN Agent for portfolio management'''
    
    def __init__(self,
                 state_size: int = 256,
                 action_size: int = 3,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 32,
                 buffer_size: int = 100000,
                 update_target_every: int = 10,
                 use_double_dqn: bool = True,
                 use_dueling: bool = True,
                 use_prioritized: bool = True,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        '''
        Initialize DQN Agent
        
        Args:
            state_size: Dimension of state space
            act# Create the main DQN agent
@"
'''Deep Q-Network Agent for Trading
Complete implementation with training and trading logic
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List, Dict
import random
from collections import deque

from .models.dqn_model import DQN, DuelingDQN, RainbowDQN
from .memory.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class TradingAction:
    '''Trading action definitions'''
    HOLD = 0
    BUY = 1
    SELL = 2
    
    @staticmethod
    def to_string(action: int) -> str:
        actions = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        return actions.get(action, 'UNKNOWN')

class DQNAgent:
    '''DQN Agent for portfolio management'''
    
    def __init__(self,
                 state_size: int = 256,
                 action_size: int = 3,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 32,
                 buffer_size: int = 100000,
                 update_target_every: int = 10,
                 use_double_dqn: bool = True,
                 use_dueling: bool = True,
                 use_prioritized: bool = True,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        '''
        Initialize DQN Agent
        
        Args:
            state_size: Dimension of state space
            action_size: Number of actions (HOLD, BUY, SELL)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            batch_size: Training batch size
            buffer_size: Replay buffer size
            update_target_every: Episodes between target network updates
            use_double_dqn: Use Double DQN
            use_dueling: Use Dueling DQN architecture
            use_prioritized: Use prioritized experience replay
            device: Computing device (cuda/cpu)
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.use_double_dqn = use_double_dqn
        self.device = torch.device(device)
        
        # Neural networks
        if use_dueling:
            self.q_network = DuelingDQN(state_size, action_size).to(self.device)
            self.target_network = DuelingDQN(state_size, action_size).to(self.device)
        else:
            self.q_network = DQN(state_size, action_size).to(self.device)
            self.target_network = DQN(state_size, action_size).to(self.device)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        if use_prioritized:
            self.memory = PrioritizedReplayBuffer(buffer_size)
            self.beta = 0.4
            self.beta_increment = 0.001
        else:
            self.memory = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.steps_done = 0
        self.episodes_done = 0
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'epsilons': [],
            'q_values': []
        }
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        '''
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Action to take
        '''
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool):
        '''Store experience in replay buffer'''
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self) -> Optional[float]:
        '''
        Train the network on a batch of experiences
        
        Returns:
            Loss value if training occurred
        '''
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        if isinstance(self.memory, PrioritizedReplayBuffer):
            experiences, weights, indices = self.memory.sample(self.batch_size, self.beta)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None
        
        # Prepare batch
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Next Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use online network to select actions
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(dim=1)[0]
            
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Calculate loss
        loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        # Update priorities if using prioritized replay
        if isinstance(self.memory, PrioritizedReplayBuffer) and indices is not None:
            td_errors = (target_q_values - current_q_values).detach().cpu().numpy()
            self.memory.update_priorities(indices, np.abs(td_errors))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update beta for prioritized replay
        if isinstance(self.memory, PrioritizedReplayBuffer):
            self.beta = min(1.0, self.beta + self.beta_increment)
        
        self.steps_done += 1
        
        # Record metrics
        self.training_history['losses'].append(loss.item())
        self.training_history['epsilons'].append(self.epsilon)
        self.training_history['q_values'].append(current_q_values.mean().item())
        
        return loss.item()
    
    def update_target_network(self):
        '''Copy weights from online to target network'''
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str):
        '''Save model and training state'''
        torch.save({
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'training_history': self.training_history
        }, filepath)
        print(f'Model saved to {filepath}')
    
    def load(self, filepath: str):
        '''Load model and training state'''
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        self.training_history = checkpoint['training_history']
        print(f'Model loaded from {filepath}')
    
    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, int]:
        '''
        Train for one episode
        
        Args:
            env: Trading environment
            max_steps: Maximum steps per episode
            
        Returns:
            Tuple of (total reward, episode length)
        '''
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Choose action
            action = self.act(state, training=True)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            self.remember(state, action, reward, next_state, done)
            
            # Learn from experience
            if len(self.memory) >= self.batch_size:
                self.learn()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update target network periodically
        self.episodes_done += 1
        if self.episodes_done % self.update_target_every == 0:
            self.update_target_network()
        
        # Record episode metrics
        self.training_history['episode_rewards'].append(total_reward)
        self.training_history['episode_lengths'].append(step + 1)
        
        return total_reward, step + 1
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        '''
        Evaluate agent performance
        
        Args:
            env: Trading environment
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary of metrics
        '''
        rewards = []
        lengths = []
        
        for _ in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            done = False
            while not done:
                action = self.act(state, training=False)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                steps += 1
            
            rewards.append(total_reward)
            lengths.append(steps)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'mean_length': np.mean(lengths)
        }



import torch.nn.functional as F

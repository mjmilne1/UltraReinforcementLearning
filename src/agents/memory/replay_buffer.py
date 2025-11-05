'''Experience Replay Buffer for DQN Agent'''

import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
import torch

# Experience tuple
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    '''Experience replay buffer for DQN training'''
    
    def __init__(self, capacity: int = 100000):
        '''Initialize replay buffer'''
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        '''Add experience to buffer'''
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        '''Sample batch of experiences'''
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        '''Check if buffer has enough samples'''
        return len(self.buffer) >= batch_size

class PrioritizedReplayBuffer:
    '''Prioritized experience replay for improved learning'''
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        '''Initialize prioritized replay buffer'''
        self.capacity = capacity
        self.alpha = alpha
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        '''Add experience with maximum priority'''
        
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # New experiences get maximum priority
        self.priorities[self.position] = self.max_priority ** self.alpha
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        '''Sample batch based on priorities'''
        n = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:n]
        probs = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(n, batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        '''Update priorities after training'''
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return len(self.buffer)

class EpisodeBuffer:
    '''Buffer for storing complete episodes'''
    
    def __init__(self, max_episodes: int = 100):
        '''Initialize episode buffer'''
        self.episodes = deque(maxlen=max_episodes)
        self.current_episode = []
        
    def add_step(self, state: np.ndarray, action: int, reward: float):
        '''Add step to current episode'''
        self.current_episode.append((state, action, reward))
    
    def end_episode(self, final_state: np.ndarray):
        '''End current episode and store it'''
        if self.current_episode:
            self.episodes.append({
                'steps': self.current_episode,
                'final_state': final_state,
                'total_reward': sum(r for _, _, r in self.current_episode),
                'length': len(self.current_episode)
            })
            self.current_episode = []
    
    def get_statistics(self) -> dict:
        '''Get episode statistics'''
        if not self.episodes:
            return {}
        
        rewards = [ep['total_reward'] for ep in self.episodes]
        lengths = [ep['length'] for ep in self.episodes]
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'mean_length': np.mean(lengths),
            'num_episodes': len(self.episodes)
        }

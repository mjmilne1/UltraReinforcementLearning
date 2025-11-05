import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

class PPOAgent:
    def __init__(self, state_size: int, action_size: int, lr: float = 3e-4):
        self.state_size = state_size
        self.action_size = action_size
        
        self.policy_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(64, action_size)
        self.critic = nn.Linear(64, 1)
        
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + 
            list(self.actor.parameters()) + 
            list(self.critic.parameters()), 
            lr=lr
        )
        
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        
    def get_action_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.policy_net(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        
        action_probs = torch.softmax(action_logits, dim=-1)
        return action_probs, value
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.get_action_prob(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
        return action
    
    def update(self, states, actions, rewards, advantages, old_probs):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        advantages = torch.FloatTensor(advantages)
        old_probs = torch.FloatTensor(old_probs)
        
        action_probs, values = self.get_action_prob(states)
        
        new_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        ratio = new_probs / old_probs
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = nn.MSELoss()(values.squeeze(), rewards)
        
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

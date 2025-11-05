import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class A2CAgent:
    '''Advantage Actor-Critic Agent'''
    
    def __init__(self, state_size: int, action_size: int, lr: float = 1e-3):
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared network
        self.shared = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Linear(128, action_size)
        
        # Critic head
        self.critic = nn.Linear(128, 1)
        
        self.optimizer = optim.Adam(
            list(self.shared.parameters()) + 
            list(self.actor.parameters()) + 
            list(self.critic.parameters()),
            lr=lr
        )
        
        self.gamma = 0.99
        
    def act(self, state: np.ndarray, training: bool = True) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        features = self.shared(state_tensor)
        action_probs = torch.softmax(self.actor(features), dim=1)
        
        if training:
            action = torch.multinomial(action_probs, 1).item()
        else:
            action = action_probs.argmax().item()
            
        return action
    
    def learn(self, state, action, reward, next_state, done):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        action_t = torch.LongTensor([action])
        reward_t = torch.FloatTensor([reward])
        done_t = torch.FloatTensor([done])
        
        # Current estimates
        features = self.shared(state_t)
        action_probs = torch.softmax(self.actor(features), dim=1)
        value = self.critic(features)
        
        # Next value
        with torch.no_grad():
            next_features = self.shared(next_state_t)
            next_value = self.critic(next_features)
            target_value = reward_t + self.gamma * next_value * (1 - done_t)
        
        # Advantage
        advantage = target_value - value
        
        # Actor loss
        log_prob = torch.log(action_probs.gather(1, action_t.unsqueeze(1)))
        actor_loss = -(log_prob * advantage.detach()).mean()
        
        # Critic loss
        critic_loss = advantage.pow(2).mean()
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

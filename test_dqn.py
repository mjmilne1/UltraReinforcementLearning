import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch

print("Testing DQN Agent Creation...")
print("=" * 60)

# Import components
from src.agents.memory.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from src.agents.models.dqn_model import DQN, DuelingDQN

# Test Replay Buffer
print("\n1. Testing Replay Buffer:")
buffer = ReplayBuffer(capacity=1000)

for i in range(100):
    state = np.random.randn(256)
    action = np.random.randint(0, 3)
    reward = np.random.randn()
    next_state = np.random.randn(256)
    done = i == 99
    buffer.push(state, action, reward, next_state, done)

print(f"   Buffer size: {len(buffer)}")
if len(buffer) >= 32:
    batch = buffer.sample(32)
    print(f"   Sampled batch size: {len(batch)}")

# Test DQN Model
print("\n2. Testing DQN Model:")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")

model = DQN(state_size=256, action_size=3)
print(f"   Model created: {type(model).__name__}")

# Test forward pass
state = torch.randn(1, 256)
with torch.no_grad():
    q_values = model(state)
    print(f"   Q-values shape: {q_values.shape}")
    print(f"   Q-values: {q_values.numpy().flatten()}")

# Test Dueling DQN
print("\n3. Testing Dueling DQN:")
dueling_model = DuelingDQN(state_size=256, action_size=3)
print(f"   Model created: {type(dueling_model).__name__}")

with torch.no_grad():
    q_values = dueling_model(state)
    print(f"   Q-values shape: {q_values.shape}")
    action = q_values.argmax(dim=1).item()
    print(f"   Selected action: {action}")

# Test Prioritized Buffer
print("\n4. Testing Prioritized Replay Buffer:")
p_buffer = PrioritizedReplayBuffer(capacity=1000)

for i in range(100):
    state = np.random.randn(256)
    action = np.random.randint(0, 3)
    reward = np.random.randn()
    next_state = np.random.randn(256)
    done = i == 99
    p_buffer.push(state, action, reward, next_state, done)

print(f"   Buffer size: {len(p_buffer)}")
if len(p_buffer) >= 32:
    batch, weights, indices = p_buffer.sample(32, beta=0.4)
    print(f"   Sampled batch size: {len(batch)}")
    print(f"   Weights shape: {weights.shape}")

print("\nAll DQN components working correctly!")

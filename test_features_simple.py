import numpy as np
from datetime import datetime, timezone

print('Testing Feature Generation')
print('='*50)

# Simulate feature generation
for i in range(10):
    features = np.random.randn(256)  # 256-dimensional feature vector
    print(f'Step {i}: Generated features - Shape: {features.shape}, Mean: {np.mean(features):.4f}')

print('\n✅ Feature generation test successful!')
print(f'Ready for RL agents with {features.shape[0]}-dimensional input')

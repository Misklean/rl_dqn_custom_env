import math
import random
import gymnasium as gym
import matplotlib
import numpy as np
import cv2
from collections import deque
from itertools import count
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from classes.DQN import DQNAgent
import torch
import matplotlib.pyplot as plt
import ale_py
from classes.CustomEnv import *

# Initialize environment
gym.register_envs(ale_py)
# env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
env = CustomEnv(render_mode='rgb_array')

# Training configuration
num_episodes = 1000  # Adjust based on your training needs
num_initial_frames = 4
video_interval = 10  # Save video every 5 episodes
video_dir = './videos'  # Directory to save videos

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.00
EPS_DECAY = 1e-4
TAU = 0.005
LR = 1e-4

# Instantiate DQN agent
agent = DQNAgent(action_size=env.action_space.n, learning_rate=LR, gamma=GAMMA, epsilon=EPS_START, epsilon_min=EPS_END,
                 epsilon_decay=EPS_DECAY, tau=TAU)

# Initialize steps_done
steps_done = 0

import matplotlib.pyplot as plt

# Initialize a list to store epsilon values
epsilon_values = []

for i_episode in range(num_episodes + 1):
    # Update epsilon
    agent.epsilon = agent.epsilon_min + (
        agent.epsilon - agent.epsilon_min
    ) * np.exp(-agent.epsilon_decay * i_episode)
    
    # Store epsilon value
    epsilon_values.append(agent.epsilon)
    
    # (Rest of your training loop code...)

# Plot the epsilon values after training
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, label='Epsilon')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay Over Episodes')
plt.legend()
plt.grid()
plt.savefig('epsilon_decay_plot.png')  # Save the plot as an image
plt.show()

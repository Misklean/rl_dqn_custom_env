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

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# Set device
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def preprocess_frame(frame):
    if len(frame.shape) == 3 and frame.shape[2] == 3:  # RGB image
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif len(frame.shape) == 3 and frame.shape[2] == 1:  # Single-channel grayscale
        gray_frame = frame.squeeze(-1)
    elif len(frame.shape) == 2:  # Already a grayscale image
        gray_frame = frame
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
    
    resized_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
    normalized_frame = resized_frame / 255.0
    return normalized_frame


def stack_frames(frames):
    return np.stack(frames, axis=0)

episode_rewards = []  # List to store rewards per episode

def plot_rewards():
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.clf()
    plt.title('Mean Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.plot(rewards_t.numpy())
    plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)

# Training configuration
num_episodes = 10000  # Adjust based on your training needs
num_initial_frames = 4
video_interval = 200  # Save video every 5 episodes
video_dir = './videos'  # Directory to save videos

# Instantiate DQN agent
agent = DQNAgent(action_size=env.action_space.n, learning_rate=LR, gamma=GAMMA, epsilon=EPS_START, epsilon_min=EPS_END,
                 epsilon_decay=EPS_DECAY, tau=TAU)

# Initialize steps_done
steps_done = 0

# Training loop
for i_episode in range(num_episodes + 1):
    if i_episode % video_interval == 0:
        # Apply RecordVideo wrapper and save the model
        env = RecordVideo(env, video_dir, episode_trigger=lambda x: x % video_interval == 0)
        
        # Save the model at the same time as recording video
        agent.save_model(filepath=f'./models/dqn_model_episode_{i_episode}.pth')
    
    state, info = env.reset()
    frame_buffer = deque(maxlen=4)
    episode_reward = 0

    # Fill the buffer with the initial frames
    for _ in range(num_initial_frames):
        frame_buffer.append(preprocess_frame(state))
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        episode_reward += reward
        state = preprocess_frame(state)

    # Stack the initial frames into the tensor before starting the episode
    state_tensor = torch.tensor(stack_frames(frame_buffer), dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():
        # Select action using the agent's policy
        action = agent.select_action(state_tensor, steps_done)  # Pass steps_done as argument

        # Take action in the environment
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        done = terminated or truncated

        # Preprocess the next frame and update the buffer
        if not done:
            frame_buffer.append(preprocess_frame(observation))
            next_state_tensor = torch.tensor(stack_frames(frame_buffer), dtype=torch.float32, device=device).unsqueeze(0)
        else:
            next_state_tensor = None

        # Store transition in replay buffer
        agent.memory.push(state_tensor, torch.tensor([[action]], device=device), next_state_tensor, torch.tensor([reward], device=device))

        # Move to the next state
        state_tensor = next_state_tensor

        # Optimize the agent's model
        if len(agent.memory) > BATCH_SIZE:
            agent.optimize_model(BATCH_SIZE)

        # Soft update of target network
        agent.update_target_network()

        # Increment steps_done
        steps_done += 1

        if done:
            episode_rewards.append(episode_reward)
            plot_rewards()
            print(f"Episode {i_episode}: Reward {episode_reward}")
    
            # # Update epsilon with exponential decay
            # agent.epsilon = agent.epsilon_min + (
            #     agent.epsilon - agent.epsilon_min
            # ) * np.exp(-agent.epsilon_decay * i_episode)

            break

    # Close and save video after recording interval episodes
    if i_episode % video_interval == 0:
        env.close()  # Ensure that video recording stops and is saved

print('Complete')
plot_rewards()
plt.ioff()
plt.show()
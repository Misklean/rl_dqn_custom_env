import pygame
import torch
import numpy as np
from classes.CustomEnv import CustomEnv
from classes.DQN import DQNAgent
from config import *
from collections import deque
import cv2
from itertools import count
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
# Set device
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Parameters
n_record_episodes = 10  # Number of episodes to play manually
max_steps = 500        # Max steps per episode
batch_size = 64         # Batch size for training

# Parameters
num_initial_frames = 4  # Number of frames to stack initially

video_folder = "./videos"
video_interval = 50

def plot_rewards(rewards, video_interval, save_path='agent_rewards_plot.png'):
    plt.figure(figsize=(10, 6))

    moving_avg = [
        np.mean(rewards[max(0, j - video_interval):j + 1])
        for j in range(len(rewards))
    ]
    
    plt.plot(moving_avg, alpha=0.8)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward (Last Video Interval)')
    plt.title('Agent Rewards')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()  # Close the figure to avoid showing it interactively

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

def play_and_train(env, agent, n_episodes, max_steps):
    """Record gameplay data while playing manually and train the model after each episode."""
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        frame_buffer = deque(maxlen=4)
        terminated = False
        step = 0

        # Fill the buffer with the initial frames
        for _ in range(num_initial_frames):
            frame_buffer.append(preprocess_frame(state))
            state, reward, terminated, truncated, info = env.step(env.action_space.sample())
            state = preprocess_frame(state)
            step += 1

        # Stack the initial frames into the tensor before starting the episode
        state_tensor = torch.tensor(stack_frames(frame_buffer), dtype=torch.float32, device=device).unsqueeze(0)

        print(f"Playing and training episode {episode + 1}/{n_episodes}")

        while not terminated and step < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            # Determine action based on key presses
            if pygame.key.get_pressed()[pygame.K_UP]:
                action = 1
            else:
                action = 0

            # Take action in the environment
            observation, reward, terminated, truncated, _ = env.step(action)
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

            env.render() 
            step += 1

        # Optimize the model with the new transitions
        agent.optimize_model(batch_size)

        # Update the target network periodically
        agent.update_target_network()
        print(f"Episode {episode + 1}: Model updated")

def train_dqn(env, agent, num_episodes=500):
    # Initialize steps_done
    steps_done = 0
    episode_rewards = []

    # Training loop
    for i_episode in range(num_episodes + 1):
            # Save the model at the same time as recording video
            # agent.save_model(filepath=f'./models/dqn_model_episode_{i_episode}.pth')
        
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
            if len(agent.memory) > batch_size:
                agent.optimize_model(batch_size)

            # Soft update of target network
            agent.update_target_network()

            # Increment steps_done
            steps_done += 1

            if done:
                episode_rewards.append(episode_reward)
                plot_rewards(episode_rewards, video_interval)
                print(f"Episode {i_episode}: Reward {episode_reward}")
        
                # # Update epsilon with exponential decay
                # agent.epsilon = agent.epsilon_min + (
                #     agent.epsilon - agent.epsilon_min
                # ) * np.exp(-agent.epsilon_decay * i_episode)

                break

if __name__ == "__main__":
    env = CustomEnv(render_mode="human")
    agent = DQNAgent(action_size=env.action_space.n)

    # Step 1: Play and train directly on the data
    play_and_train(env, agent, n_episodes=n_record_episodes, max_steps=max_steps)

    env = CustomEnv(render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda episode: episode % video_interval == 0, disable_logger=True)

    train_dqn(env, agent, num_episodes=1000)
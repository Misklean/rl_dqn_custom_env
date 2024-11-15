import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import threading
import cv2
from classes.CustomEnv import CustomEnv
import torch.nn.functional as F
import pygame
from queue import Queue
from gymnasium.wrappers import RecordVideo
import time

from config import *

model_lock = threading.Lock()
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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

class GlobalDQN(nn.Module):
    def __init__(self, n_frames, n_actions):
        super(GlobalDQN, self).__init__()
        self.conv1 = nn.Conv2d(n_frames, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Assuming input resolution is 84x84
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the convolutional output
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = Queue(maxsize=capacity)
        self.lock = threading.Lock()  # Lock for safe memory access

    def push(self, *args):
        with self.lock:
            if not self.memory.full():
                self.memory.put(Transition(*args))

    def sample(self, batch_size):
        with self.lock:
            sampled = []
            while not self.memory.empty() and len(sampled) < batch_size:
                sampled.append(self.memory.get())
            return sampled

    def __len__(self):
        return self.memory.qsize()

class Agent(threading.Thread):
    def __init__(self, id, global_dqn, target_dqn, memory, env, hyperparams, record_interval=50, update_interval=5):
        super(Agent, self).__init__()
        self.id = id
        self.global_dqn = global_dqn
        self.target_dqn = target_dqn
        self.env = env
        self.hyperparams = hyperparams
        self.epsilon = hyperparams['epsilon']
        self.epsilon_min = hyperparams['epsilon_min']
        self.epsilon_decay = hyperparams['epsilon_decay']
        self.gamma = hyperparams['gamma']
        self.global_optimizer = optim.AdamW(self.global_dqn.parameters(), lr=0.001, amsgrad=True)
        self.global_memory = memory
        self.record_interval = record_interval
        self.update_interval = update_interval
        self.local_steps = 0

    def select_action(self, state):
        if np.random.rand() > self.epsilon:
            with torch.no_grad():
                return self.global_dqn(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.env.action_space.n)]], device=device, dtype=torch.long)

    def run(self):
        for i_episode in range(self.hyperparams['num_episodes'] + 1):
            state, info = self.env.reset()
            state = preprocess_frame(state)
            frame_buffer = deque([state] * num_initial_frames, maxlen=num_initial_frames)
            state_tensor = torch.tensor(stack_frames(frame_buffer), dtype=torch.float32, device=device).unsqueeze(0)
            episode_reward = 0
            clock = pygame.time.Clock()

            while True:
                action = self.select_action(state_tensor)
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_reward += reward
                done = terminated or truncated

                if not done:
                    next_state = preprocess_frame(next_state)
                    frame_buffer.append(next_state)
                    next_state_tensor = torch.tensor(stack_frames(frame_buffer), dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    next_state_tensor = None

                self.global_memory.push(state_tensor, action, next_state_tensor, torch.tensor([reward], device=device))
                state_tensor = next_state_tensor

                if done:
                    print(f"Agent {self.id} - Episode {i_episode}: Reward {episode_reward} Epsilon {self.epsilon}")
                    self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

                    # Save the model if the episode is a video interval
                    if i_episode % video_interval == 0:
                        model_path = f"./models/model_episode_{i_episode}_agent_{self.id}.pth"
                        torch.save(self.global_model.state_dict(), model_path)
                        print(f"Model saved at {model_path}")
                    break

                if len(self.global_memory) >= self.hyperparams['batch_size']:
                    if self.local_steps % self.update_interval == 0:
                        self.optimize_global_model()

                self.local_steps = self.local_steps + 1


            self.update_target_network()

    def optimize_global_model(self):
        with model_lock:

            if len(self.global_memory) < self.hyperparams['batch_size']:
                return

            transitions = self.global_memory.sample(self.hyperparams['batch_size'])
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
            state_batch = torch.cat(batch.state).to(device)
            action_batch = torch.cat(batch.action).to(device)
            reward_batch = torch.cat(batch.reward).to(device)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

            # Ensure the state-action values are not modified in-place
            state_action_values = self.global_dqn(state_batch).gather(1, action_batch)

            # Initialize next_state_values tensor without in-place modification
            next_state_values = torch.zeros(self.hyperparams['batch_size'], device=device)

            # Use detach() to avoid tracking gradients
            next_state_actions = self.global_dqn(non_final_next_states).max(1)[1].unsqueeze(1)
            next_state_values[non_final_mask] = self.target_dqn(non_final_next_states).gather(1, next_state_actions).detach().squeeze()

            # Calculate expected state-action values
            expected_state_action_values = ((next_state_values * self.gamma) + reward_batch).to(device)

            # Calculate loss (without in-place modification)
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            # Zero gradients before backward pass
            self.global_optimizer.zero_grad()

            # Backpropagate the loss
            loss.backward()

            # Update model weights
            self.global_optimizer.step()

    def update_target_network(self):
        self.target_dqn.load_state_dict(self.global_dqn.state_dict())

def train_multiple_agents():
    torch.autograd.set_detect_anomaly(True)

    # Initialize environments, DQNs, and memory
    envs = [
        RecordVideo(
            CustomEnv(max_steps=MAX_STEPS, render_mode="rgb_array"),
            video_folder=f"{video_dir}/agent_{i}",
            episode_trigger=lambda episode: episode % video_interval == 0,
            disable_logger=True
        ) for i in range(num_agents)
    ]

    global_dqn = GlobalDQN(num_initial_frames, N_DISCRETE_ACTIONS).to(device)
    target_dqn = GlobalDQN(num_initial_frames, N_DISCRETE_ACTIONS).to(device)
    target_dqn.load_state_dict(global_dqn.state_dict())
    memory = ReplayMemory(50000)

    hyperparams_list = [
        {"epsilon": 1.0, "epsilon_min": 0.05, "epsilon_decay": (1 - 0.05) / num_episodes, "gamma": 0.99, "num_episodes": num_episodes, "batch_size": 64},
        {"epsilon": 1.0, "epsilon_min": 0.05, "epsilon_decay": (1 - 0.05) / (num_episodes * 1.1), "gamma": 0.99, "num_episodes": num_episodes, "batch_size": 64},
        {"epsilon": 1.0, "epsilon_min": 0.05, "epsilon_decay": (1 - 0.05) / (num_episodes * 0.9), "gamma": 0.99, "num_episodes": num_episodes, "batch_size": 64},
        {"epsilon": 1.0, "epsilon_min": 0, "epsilon_decay": (1 - 0.05) / num_episodes, "gamma": 0.99, "num_episodes": num_episodes, "batch_size": 64}
    ]

    # Initialize agents with hyperparameters
    agents = [
        Agent(i, global_dqn, target_dqn, memory, envs[i], hyperparams_list[i % len(hyperparams_list)])
        for i in range(num_agents)
    ]

    start_time = time.time()
    
    # Start agent threads
    for agent in agents:
        agent.start()
    
    # Join all agents
    for agent in agents:
        agent.join()

    end_time = time.time()  # End timing
    total_time = end_time - start_time  # Calculate elapsed time
    print(f"Training completed in {total_time:.2f} seconds")

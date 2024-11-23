import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import torch.nn.functional as F

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(64 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, tau=0.001):
        self.q_network = DQN(action_size).to(device)  # Define q_network and move to device
        self.target_q_network = DQN(action_size).to(device)  # Define target network and move to device
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=learning_rate, amsgrad=True)
        self.memory = ReplayMemory(50000)  # Set capacity of replay buffer
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.action_size = action_size
        self.losses = []
        self._get_eps = lambda n_steps: self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(
            -1.0 * n_steps / self.epsilon_decay
        )


    def select_action(self, state, steps_done):
        if np.random.rand() > self.epsilon:
            with torch.no_grad():
                return self.q_network(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

    def optimize_model(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Sample a batch of transitions from the replay buffer
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)

        # Convert batch data to tensors
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)

        # Compute Q(s_t, a) - the Q values for the actions taken
        state_action_values = self.q_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states with Double Q-learning
        # Use the main Q-network to select the best action for each next state
        next_state_actions = self.q_network(non_final_next_states).max(1)[1].unsqueeze(1)

        # Evaluate that action using the target network
        next_state_values = torch.zeros(batch_size, device=device)
        next_state_values[non_final_mask] = self.target_q_network(non_final_next_states).gather(1, next_state_actions).squeeze()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_target_network(self):
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, filepath="dqn_model.pth"):
        torch.save(self.q_network.state_dict(), filepath)
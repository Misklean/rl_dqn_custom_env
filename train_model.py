import gymnasium as gym
import matplotlib
import numpy as np
from collections import deque
from itertools import count
from classes.DQN import DQNAgent
import torch
import matplotlib.pyplot as plt
from classes.CustomEnv import CustomEnv
from gymnasium.wrappers import RecordVideo

from config import *

# Set device
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


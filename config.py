import numpy as np

# Define constants
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 400
CELL_SIZE = 20  # Size of the character
ENV_WIDTH, ENV_HEIGHT = 20 * CELL_SIZE, 20 * CELL_SIZE  # Size of the entire environment (can be bigger than screen)
N_DISCRETE_ACTIONS = 4  # up, down, left, right
FPS = 30
SPEED = 5  # Speed of movement
MAP_WIDTH, MAP_HEIGHT = ENV_WIDTH // CELL_SIZE, ENV_HEIGHT // CELL_SIZE
NB_LEVELS = 1
MAX_STEPS = 500

# Training configuration
num_episodes = 1000  # Adjust based on your training needs
num_initial_frames = 4
video_interval = 50  # Save video every 5 episodes
video_dir = './media/videos'  # Directory to save videos
n_last_episodes = 50
sync_interval = 10

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = (EPS_START - EPS_END) / num_episodes
TAU = 0.005
LR = 2.5e-4
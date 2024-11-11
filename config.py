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
MAX_STEPS = 300

# Training configuration
num_episodes = 1000  # Adjust based on your training needs
num_initial_frames = 4
video_interval = 50  # Save video every 5 episodes
video_dir = './media/videos'  # Directory to save videos

# Hyperparameters
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = (EPS_END / EPS_START) ** (1 / num_episodes)
TAU = 0.001
LR = 2.5e-4
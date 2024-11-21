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
NB_GREEN_CELLS = 10

# Training configuration
num_episodes = 1000  # Adjust based on your training needs
num_initial_frames = 4
video_interval = 20  # Save video every 5 episodes
video_dir = './media/videos'  # Directory to save videos
sync_interval = 10
num_agents = 1
last_final_episode = -1

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
TAU = 0.005
LR = 2.5e-4
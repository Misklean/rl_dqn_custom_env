# Define constants
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 400
SIZE = 20
COLOR = (255, 0, 0)
PLAYER_X = SCREEN_WIDTH // 5 - SIZE // 2
PLAYER_Y = SCREEN_HEIGHT - 20
SPEED = 5
FPS = 30
GRAVITY = 0.5
PLATFORM_HEIGHT = 20

MAX_STEPS = 500

# Parameters
NB_EPISODES = 1000
n_record_episodes = 10  # Number of episodes to play manually
max_steps = 500        # Max steps per episode
batch_size = 64         # Batch size for training
num_initial_frames = 4  # Number of frames to stack initially

video_folder = "./results/videos"
video_interval = 50

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.00
EPS_DECAY = 1e-4
TAU = 0.005
LR = 1e-4
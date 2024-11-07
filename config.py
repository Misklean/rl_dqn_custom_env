# Define constants
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 400
ENV_WIDTH, ENV_HEIGHT = 30 * 20, 30 * 20  # Size of the entire environment (can be bigger than screen)
N_DISCRETE_ACTIONS = 4  # up, down, left, right
FPS = 30
SPEED = 5  # Speed of movement
CELL_SIZE = 20  # Size of the character
MAP_WIDTH, MAP_HEIGHT = ENV_WIDTH // CELL_SIZE, ENV_HEIGHT // CELL_SIZE
NB_LEVELS = 10
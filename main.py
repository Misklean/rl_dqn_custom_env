import pygame
import gym
from gym import spaces
import numpy as np
import random

# Define constants
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 400
ENV_WIDTH, ENV_HEIGHT = 800, 800  # Size of the entire environment (can be bigger than screen)
N_DISCRETE_ACTIONS = 4  # up, down, left, right
FPS = 30
SPEED = 5  # Speed of movement
CELL_SIZE = 20  # Size of the character
MAP_WIDTH, MAP_HEIGHT = ENV_WIDTH // CELL_SIZE, ENV_HEIGHT // CELL_SIZE

class CustomEnv(gym.Env):
    """Environment with a character that can move around a procedurally generated map."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Initial position of the agent
        self.agent_pos = np.array([ENV_WIDTH // 2, ENV_HEIGHT // 2])  # Ensure this is a NumPy array

        # Camera offset
        self.camera_x = 0
        self.camera_y = 0

        # Generate map using automaton
        self.map = self.generate_map()

    def generate_map(self):
        """Generate a 2D map using cellular automaton."""
        map_grid = np.random.choice([0, 1], size=(MAP_HEIGHT, MAP_WIDTH), p=[0.5, 0.5])  # 0 is walkable, 1 is a wall

        for _ in range(4):  # Apply automaton rules 4 times to smooth the map
            new_map = np.copy(map_grid)
            for y in range(1, MAP_HEIGHT - 1):
                for x in range(1, MAP_WIDTH - 1):
                    # Count the number of walls around the current cell
                    wall_count = np.sum(map_grid[y - 1:y + 2, x - 1:x + 2]) - map_grid[y, x]

                    # Apply automaton rules to create a more structured map
                    if wall_count > 4:
                        new_map[y, x] = 1  # Become a wall
                    elif wall_count < 4:
                        new_map[y, x] = 0  # Become walkable

            map_grid = new_map

        return map_grid

    def step(self, action):
        # Store the original position as a NumPy array
        original_pos = self.agent_pos.copy()

        # Create a Rect for the agent
        agent_rect = pygame.Rect(self.agent_pos[0], self.agent_pos[1], CELL_SIZE, CELL_SIZE)

        # Calculate intended new position based on action
        if action == 0:  # up
            intended_pos = (original_pos[0], original_pos[1] - SPEED)
        elif action == 1:  # down
            intended_pos = (original_pos[0], original_pos[1] + SPEED)
        elif action == 2:  # left
            intended_pos = (original_pos[0] - SPEED, original_pos[1])
        elif action == 3:  # right
            intended_pos = (original_pos[0] + SPEED, original_pos[1])

        # Boundaries to prevent going out of the environment
        intended_pos = (
            np.clip(intended_pos[0], 0, ENV_WIDTH - CELL_SIZE),
            np.clip(intended_pos[1], 0, ENV_HEIGHT - CELL_SIZE)
        )

        # Create a new Rect for the intended position of the agent
        intended_rect = pygame.Rect(intended_pos[0], intended_pos[1], CELL_SIZE, CELL_SIZE)

        # Create Rects for walls based on the map
        walls = []
        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                if self.map[y, x] == 1:  # If it's a wall
                    wall_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    walls.append(wall_rect)

        # Check for collisions with walls
        for wall_rect in walls:
            if intended_rect.colliderect(wall_rect):
                # Snap to the edge of the wall based on the action
                if action == 0:  # up
                    intended_pos = (intended_pos[0], wall_rect.bottom)  # Snap to the bottom edge of the wall
                elif action == 1:  # down
                    intended_pos = (intended_pos[0], wall_rect.top - CELL_SIZE)  # Snap to the top edge of the wall
                elif action == 2:  # left
                    intended_pos = (wall_rect.right, intended_pos[1])  # Snap to the right edge of the wall
                elif action == 3:  # right
                    intended_pos = (wall_rect.left - CELL_SIZE, intended_pos[1])  # Snap to the left edge of the wall
                break  # Exit the loop once a collision is found

        # Update the agent's position to the intended position
        self.agent_pos = np.array(intended_pos)  # Ensure this is a NumPy array

        # Update the camera's position based on the agent's position
        self.camera_x = np.clip(self.agent_pos[0] - SCREEN_WIDTH // 2, 0, ENV_WIDTH - SCREEN_WIDTH)
        self.camera_y = np.clip(self.agent_pos[1] - SCREEN_HEIGHT // 2, 0, ENV_HEIGHT - SCREEN_HEIGHT)

        return self.get_observation(), 0, False, {}


    def reset(self):
        # Reset the agent position to the center of the environment
        self.agent_pos = np.array([ENV_WIDTH // 2, ENV_HEIGHT // 2])

        # Reset the camera position
        self.camera_x = 0
        self.camera_y = 0

        return self.get_observation()

    def render(self, mode='human'):
        # Clear the screen
        self.screen.fill((0, 0, 0))

        # Draw the map relative to the camera
        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                rect = pygame.Rect(x * CELL_SIZE - self.camera_x, y * CELL_SIZE - self.camera_y, CELL_SIZE, CELL_SIZE)
                color = (0, 0, 0) if self.map[y, x] == 1 else (255, 255, 255)  # Black for walls, white for walkable
                pygame.draw.rect(self.screen, color, rect)

        # Draw the agent (red square)
        pygame.draw.rect(self.screen, (255, 0, 0), 
                         pygame.Rect(self.agent_pos[0] - self.camera_x, self.agent_pos[1] - self.camera_y, CELL_SIZE, CELL_SIZE))

        # Update the display
        pygame.display.flip()
        self.clock.tick(FPS)

    def get_observation(self):
        # Create an observation array representing the screen
        observation = pygame.surfarray.array3d(pygame.display.get_surface())  # Capture the screen as a numpy array
        observation = np.transpose(observation, (1, 0, 2))  # Transpose to match height x width x 3 shape
        return observation

# Main function
if __name__ == "__main__":
    env = CustomEnv()
    obs = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Move the agent based on key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            env.step(0)  # up
        if keys[pygame.K_DOWN]:
            env.step(1)  # down
        if keys[pygame.K_LEFT]:
            env.step(2)  # left
        if keys[pygame.K_RIGHT]:
            env.step(3)  # right

        # Render the environment
        env.render()

    pygame.quit()
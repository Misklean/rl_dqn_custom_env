import pygame
import numpy as np
import random
import gymnasium
from gymnasium import spaces
import cv2
from pygame.surfarray import array3d

from classes.MapManager import *
from classes.Agent import *

from config import *

class CustomEnv(gymnasium.Env):
    """Environment with a character that can move around a procedurally generated map."""
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps=MAX_STEPS, render_mode="human", rank=1):
        super(CustomEnv, self).__init__()
        pygame.init()
        pygame.display.set_caption("Custom Env")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(2)

        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT

        if render_mode == "human":
            # Only initialize a display window if render_mode is 'human'
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        elif render_mode == "rgb_array":
            # Create an off-screen surface for capturing frames
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.reset()

    def step(self, action):
        # Handle jumping behavior when the action is 1 (pressing SPACE to jump)
        if action == 1 and not self.is_jump:  # Only jump if on the ground
            self.velocity_y = -self.jump_count  # Set upward velocity
            self.is_jump = True

        # Apply gravity to the character's vertical velocity
        self.velocity_y += self.gravity  # Gravity pulls the player down
        self.player_y += self.velocity_y  # Update the vertical position based on the velocity

        # Prevent the cube from going below the platform (ground)
        if self.player_y + self.player_size >= self.screen_height - self.platform_height:
            self.player_y = self.screen_height - self.player_size - self.platform_height  # Reset to the ground level
            self.velocity_y = 0  # Stop falling
            self.is_jump = False

        # Prevent the cube from going off-screen horizontally
        self.player_x = max(0, min(self.player_x, self.screen_width - self.player_size))

        # Move obstacles
        for obstacle in self.obstacles:
            obstacle["x"] -= self.obstacle_speed

        # Remove obstacles that are off-screen
        self.obstacles = [o for o in self.obstacles if o["x"] + o["width"] > 0]

        # Increment spawn timer
        self.spawn_timer += 1

        # Spawn new obstacles if the delay is met
        if self.spawn_timer >= self.min_spawn_delay and random.random() < self.obstacle_spawn_chance:
            self.spawn_obstacle()
            self.spawn_timer = 0  # Reset timer after spawning an obstacle

        # Compute rewards
        terminated, truncated, reward = self.compute_rewards()

        info = {}
        self.current_step += 1

        return self.get_observation(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_x = PLAYER_X
        self.player_y = PLAYER_y
        self.player_size = SIZE
        self.player_speed = SPEED
        self.velocity_y = 0
        self.player_color = COLOR
        self.gravity = GRAVITY
        self.is_jump = False
        self.jump_count = 10
        self.current_step = 0

        # Obstacle settings
        self.obstacles = []  # List to hold obstacles
        self.obstacle_speed = 5  # Speed at which obstacles move
        self.obstacle_spawn_chance = 0.2  # Probability for additional randomness
        self.min_spawn_delay = 50  # Minimum delay in frames before spawning next obstacle
        self.spawn_timer = 0  # Tracks steps since last obstacle spawned

        self.platform_height = PLATFORM_HEIGHT
        
        # Return game state without rendering surfaces
        return self.get_observation(), {}

    def render(self):
        # Select the rendering target based on the mode
        mode = self.render_mode

        # Fill the screen with black
        self.screen.fill((0, 0, 0))

        # Draw the green platform at the bottom
        pygame.draw.rect(self.screen, (255, 255, 255), (0, self.screen_height - self.platform_height, self.screen_width, self.platform_height))

        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, (0, 255, 0), (obstacle["x"], obstacle["y"], obstacle["width"], obstacle["height"]))

        # Draw the red cube
        pygame.draw.rect(self.screen, self.player_color, (self.player_x, self.player_y, self.player_size, self.player_size))

        # Handle different render modes
        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(FPS)

        if mode == 'rgb_array':
            # Capture the surface and return as an array (picklable)
            screen_array = array3d(self.screen)
            return np.transpose(screen_array, (1, 0, 2))  # Transpose for compatibility

    def get_observation(self):
        # Capture the screen as an RGB array (using NumPy for pickling compatibility)
        observation = self.screen_capture()
        return observation.astype(np.uint8)

    def screen_capture(self):
        # Capture the screen from `self.screen`, whether it's off-screen or the display surface
        screen_array = array3d(self.screen)
        return np.transpose(screen_array, (1, 0, 2))  # Transpose if needed to match the expected format

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def spawn_obstacle(self):
        obstacle_width = random.randint(20, 50)
        obstacle_height = random.randint(20, 50)
        obstacle_x = self.screen_width
        obstacle_y = self.screen_height - self.platform_height - obstacle_height
        self.obstacles.append({
            "x": obstacle_x,
            "y": obstacle_y,
            "width": obstacle_width,
            "height": obstacle_height
        })

    def compute_rewards(self):
        reward = 0
        passed_obstacles = 0

        # Check for collision with obstacles
        player_rect = pygame.Rect(self.player_x, self.player_y, self.player_size, self.player_size)
        for obstacle in self.obstacles:
            obstacle_rect = pygame.Rect(obstacle["x"], obstacle["y"], obstacle["width"], obstacle["height"])
            if player_rect.colliderect(obstacle_rect):
                return True, False, -5  # terminated, truncated, negative reward for collision

        # Check if maximum steps are reached
        if self.current_step >= self.max_steps:
            return False, True, reward  # terminated, truncated, reward for surviving till max steps

        # Check if the player has passed any obstacles
        for obstacle in self.obstacles:
            # If the player has passed the obstacle (the player's x position is greater than the obstacle's x position)
            if self.player_x > obstacle["x"] + obstacle["width"]:
                passed_obstacles += 1

        # Give the player a reward for each obstacle they have passed
        reward += passed_obstacles

        # Return the result
        return False, False, reward

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

        # Set the terminated and truncated flags
        terminated = False
        truncated = False
        reward = 0  # Placeholder for reward, modify as needed
        info = {}  # Placeholder for additional info, modify as needed

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
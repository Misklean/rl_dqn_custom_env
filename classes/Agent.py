import pygame
import numpy as np
import random

from config import *

class Agent():
    """Manager for the map."""

    def __init__(self, map_manager):
        self.map_manager = map_manager
        # Set the agent to a random walkable (white) cell
        self.agent_pos = self.get_random_walkable_position()

        # Update the camera's position based on the agent's position
        self.update_camera()

    def update_camera(self):
        """Update the camera's position to follow the agent."""
        self.camera_x = np.clip(self.agent_pos[0] - SCREEN_WIDTH // 2, 0, ENV_WIDTH - SCREEN_WIDTH)
        self.camera_y = np.clip(self.agent_pos[1] - SCREEN_HEIGHT // 2, 0, ENV_HEIGHT - SCREEN_HEIGHT)

    def get_random_walkable_position(self):
        """Find a random walkable (white) cell for the agent to spawn."""
        while True:
            x = random.randint(0, MAP_WIDTH - 1)
            y = random.randint(0, MAP_HEIGHT - 1)
            if self.map_manager.map[y, x] == 0:  # Check if it's a walkable cell (white)
                return np.array([x * CELL_SIZE, y * CELL_SIZE])
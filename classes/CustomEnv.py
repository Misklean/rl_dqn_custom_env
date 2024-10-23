import pygame
import gym
from gym import spaces
import numpy as np
import random

from classes.MapManager import *
from classes.Agent import *

from config import *

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

        # Generate map using automaton
        self.map_manager = MapManager()
        self.map_manager.set_map(0)

        self.agent = Agent(self.map_manager)

    def step(self, action):
        # Store the original position as a NumPy array
        original_pos = self.agent.agent_pos.copy()

        # Create a Rect for the agent
        agent_rect = pygame.Rect(self.agent.agent_pos[0], self.agent.agent_pos[1], CELL_SIZE, CELL_SIZE)

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

        # Check for collisions with walls
        for wall_rect in self.map_manager.walls:
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
        self.agent.agent_pos = np.array(intended_pos)  # Ensure this is a NumPy array

        # Update the camera's position based on the agent's new position
        self.agent.update_camera()

        # Check if the agent collides with any special green cell
        for special_rect in self.map_manager.special_cells:
            if intended_rect.colliderect(special_rect):
                print("You touched the green cell!")

        return self.get_observation(), 0, False, {}

    def reset(self):
        # Reset the agent position to a random walkable cell
        self.agent.agent_pos = self.agent.get_random_walkable_position()

        # Reset the camera position
        self.agent.update_camera()

        return self.get_observation()


    def render(self, mode='human'):
        # Clear the screen
        self.screen.fill((0, 0, 0))

        # Draw the map relative to the camera
        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                # Calculate the rectangle's position relative to the camera
                rect = pygame.Rect(x * CELL_SIZE - self.agent.camera_x, y * CELL_SIZE - self.agent.camera_y, CELL_SIZE, CELL_SIZE)
                
                # Determine the color based on the map cell value
                if self.map_manager.map[y, x] == 1:
                    color = (0, 0, 0)  # Wall: black
                elif self.map_manager.map[y, x] == -1:
                    color = (0, 255, 0)  # Special cell: green
                else:
                    color = (255, 255, 255)  # Walkable cell: white
                    
                pygame.draw.rect(self.screen, color, rect)

        # Draw the agent (red square)
        pygame.draw.rect(self.screen, (255, 0, 0), 
                        pygame.Rect(self.agent.agent_pos[0] - self.agent.camera_x, 
                                    self.agent.agent_pos[1] - self.agent.camera_y, 
                                    CELL_SIZE, CELL_SIZE))

        # Update the display
        pygame.display.flip()
        self.clock.tick(FPS)

    def get_observation(self):
        # Create an observation array representing the screen
        observation = pygame.surfarray.array3d(pygame.display.get_surface())  # Capture the screen as a numpy array
        observation = np.transpose(observation, (1, 0, 2))  # Transpose to match height x width x 3 shape
        return observation
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

    def __init__(self, max_steps=MAX_STEPS, render_mode="human"):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.uint8)

        pygame.init()
        self.render_mode = render_mode
        self.max_steps = max_steps
        # self.clock = pygame.time.Clock()

        self.green_cell_img = pygame.image.load("./images/green_cell_pattern.png")
        self.green_cell_img = pygame.transform.scale(self.green_cell_img, (CELL_SIZE, CELL_SIZE))

        if render_mode == "human":
            # Only initialize a display window if render_mode is 'human'
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        elif render_mode == "rgb_array":
            # Create an off-screen surface for capturing frames
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.reset()

    def step(self, action):
        # Store the original position as a NumPy array
        original_pos = self.agent.agent_pos.copy()
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

        # Track if the agent is unable to move due to a wall collision
        stuck_against_wall = False

        # Check for collisions with walls
        for wall_rect in self.map_manager.walls:
            if intended_rect.colliderect(wall_rect):
                # If there's a collision, prevent the move and mark as "stuck"
                stuck_against_wall = True
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

        # Update the agent's position
        self.agent.agent_pos = np.array(intended_pos)

        self.agent.update_camera()

        reward, terminated, truncated = self.compute_reward(stuck_against_wall, intended_rect, original_pos)

        # Return observation, reward, terminated, truncated, and info
        return self.get_observation(), reward, terminated, truncated, {}
    
    def compute_reward(self, stuck_against_wall, intended_rect, original_pos):
        # Initialize reward
        reward = 0  # Base penalty for every step

        # Check if the agent collides with any special green cell
        reached_final_cell = False
        for special_rect in self.map_manager.special_cells:
            if intended_rect.colliderect(special_rect) and not self.green_cell_touched:
                reward += 500
                print("You touched the green cell!")
                reached_final_cell = True
                self.green_cell_touched = True
                no_more_map = self.next_map()

        # Increment step counter
        self.step_count += 1

        # Check termination conditions
        terminated = reached_final_cell and no_more_map # True if it's the final green cell
        truncated = self.step_count >= self.max_steps  # True if step limit reached

        self.update_visibility()

        # Exploration reward based on the proportion of discovered cells
        explored_cells = np.count_nonzero(self.visibility_grid)  # Count cells with True values

        # Reward agent for exploring new areas
        if explored_cells > self.last_explored_cell:
            reward += 1  # Bonus reward for each new cell discovered

        self.last_explored_cell = explored_cells

        # Penalize the agent only if it’s stuck (hits a wall and can't move)
        if stuck_against_wall and np.array_equal(original_pos, self.agent.agent_pos):
            reward -= 1  # Penalty for getting stuck

        return reward, terminated, truncated


    def next_map(self):
        """Switch to the next map or declare the player has won the game."""
        if self.map_manager.current_map_index < len(self.map_manager.maps) - 1:
            self.green_cell_touched = False
            
            # Set the new map
            self.map_manager.set_map(self.map_manager.current_map_index + 1)
            
            # Reset agent position on the new map
            self.agent = Agent(self.map_manager)
            self.agent.agent_pos = self.agent.get_random_walkable_position()
            self.agent.update_camera()

            # Initialize the visibility grid, marking walls as "seen"
            self.visibility_grid = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=bool)
            for y in range(MAP_HEIGHT):
                for x in range(MAP_WIDTH):  
                    if self.map_manager.map[y, x] == 1:
                        self.visibility_grid[y, x] = True  # Mark wall cells as explored

            self.update_visibility()

            # Track explored cells count (walls only counted initially)
            self.last_explored_cell = np.count_nonzero(self.visibility_grid)
            return False
        else:
            return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        # Initialize map manager and agent without involving surfaces
        self.map_manager = MapManager()
        self.map_manager.set_map(0)

        self.agent = Agent(self.map_manager)

        # Initialize agent position without surfaces
        self.agent.agent_pos = self.agent.get_random_walkable_position()

        # Initialize visibility grid and track explored cells
        self.visibility_grid = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=bool)
        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):  
                if self.map_manager.map[y, x] == 1:
                    self.visibility_grid[y, x] = True  # Mark wall cells as explored

        self.update_visibility()

        self.last_explored_cell = np.count_nonzero(self.visibility_grid)
        self.green_cell_touched = False

        # Return game state without rendering surfaces
        return self.get_observation(), {}
        
    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def render(self):
        # Select the rendering target based on the mode
        render_surface = self.screen
        mode = self.render_mode

        # Rendering logic (unchanged, but avoids sharing surfaces)
        render_surface.fill((0, 0, 0))  # Clear with black

        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                rect = pygame.Rect(
                    x * CELL_SIZE - self.agent.camera_x,
                    y * CELL_SIZE - self.agent.camera_y,
                    CELL_SIZE,
                    CELL_SIZE
                )

                if self.visibility_grid[y, x]:
                    if self.map_manager.map[y, x] == 1:
                        color = (0, 0, 0)  # Wall: black
                        pygame.draw.rect(render_surface, color, rect)
                    elif self.map_manager.map[y, x] == -1:
                        render_surface.blit(self.green_cell_img, rect)
                    else:
                        color = (255, 255, 255)  # Walkable cell: white
                        pygame.draw.rect(render_surface, color, rect)
                else:
                    color = (100, 100, 100)  # Unexplored areas: gray
                    pygame.draw.rect(render_surface, color, rect)

        # Draw the agent as a red rectangle
        pygame.draw.rect(render_surface, (255, 0, 0),
                        pygame.Rect(
                            self.agent.agent_pos[0] - self.agent.camera_x,
                            self.agent.agent_pos[1] - self.agent.camera_y,
                            CELL_SIZE, CELL_SIZE
                        ))

        # Handle different render modes
        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(FPS)

        if mode == 'rgb_array':
            # Capture the surface and return as an array (picklable)
            screen_array = array3d(render_surface)
            return np.transpose(screen_array, (1, 0, 2))  # Transpose for compatibility

    def display_current_level(self):
        """Display the current level as text on the screen with a background."""
        level_text = f"Level: {self.map_manager.current_map_index + 1}"
        text_surface = self.font.render(level_text, True, (255, 255, 255))  # White text
        text_rect = text_surface.get_rect(topleft=(10, 10))

        # Draw a black rectangle behind the text
        pygame.draw.rect(self.screen, (0, 0, 0), text_rect.inflate(10, 10))  # Inflate to add some padding

        # Render the text on top of the rectangle
        self.screen.blit(text_surface, text_rect.topleft)


    def get_observation(self):
        # Capture the screen as an RGB array (using NumPy for pickling compatibility)
        observation = self.screen_capture()
        return observation.astype(np.uint8)

    def screen_capture(self):
        # Capture the screen from `self.screen`, whether it's off-screen or the display surface
        screen_array = array3d(self.screen)
        return np.transpose(screen_array, (1, 0, 2))  # Transpose if needed to match the expected format


    def update_visibility(self):
        """Update visibility grid to reveal the area around the agent."""
        agent_x, agent_y = self.agent.agent_pos // CELL_SIZE  # Agent's position in grid units
        radius = 120 // CELL_SIZE  # Reveal radius in grid units (80 pixels in terms of cells)

        # Mark cells within the radius as visible
        for y in range(max(0, agent_y - radius), min(MAP_HEIGHT, agent_y + radius + 1)):
            for x in range(max(0, agent_x - radius), min(MAP_WIDTH, agent_x + radius + 1)):
                distance = np.sqrt((x - agent_x) ** 2 + (y - agent_y) ** 2)
                if distance * CELL_SIZE <= 120:  # If within reveal radius
                    self.visibility_grid[y, x] = True

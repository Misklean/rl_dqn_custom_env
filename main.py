import pygame
import gym
from gym import spaces
import numpy as np
import random

# Define constants
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 400
ENV_WIDTH, ENV_HEIGHT = 50 * 20, 50 * 20  # Size of the entire environment (can be bigger than screen)
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

        # Generate map using automaton
        self.generate_map()

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
            if self.map[y, x] == 0:  # Check if it's a walkable cell (white)
                return np.array([x * CELL_SIZE, y * CELL_SIZE])


    def generate_map(self):
        """Generate a 2D map using cellular automaton and create Rects for walls."""
        map_grid = np.random.choice([0, 1], size=(MAP_HEIGHT, MAP_WIDTH), p=[0.5, 0.5])  # 0 is walkable, 1 is a wall

        # Apply cellular automaton rules multiple times to smooth the map
        for _ in range(5):
            new_map = np.copy(map_grid)
            for y in range(1, MAP_HEIGHT - 1):
                for x in range(1, MAP_WIDTH - 1):
                    wall_count = np.sum(map_grid[y - 1:y + 2, x - 1:x + 2]) - map_grid[y, x]
                    if wall_count > 4:
                        new_map[y, x] = 1  # Become a wall
                    elif wall_count < 4:
                        new_map[y, x] = 0  # Become walkable
            map_grid = new_map

        # Set borders to be walls
        map_grid[0, :] = 1  # Top border
        map_grid[-1, :] = 1  # Bottom border
        map_grid[:, 0] = 1  # Left border
        map_grid[:, -1] = 1  # Right border

        # Ensure all white regions (rooms) are connected
        self.connect_rooms(map_grid)

        # Create Rects for walls
        walls = []
        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                if map_grid[y, x] == 1:  # If it's a wall
                    wall_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    walls.append(wall_rect)

        self.walls = walls  # Store the walls for future use
        self.map = map_grid

        self.save_map_as_image("debug_map.png")

    def save_map_as_image(self, filename="generated_map.png"):
        """Save the generated map as an image file."""
        # Create a surface to draw the map
        map_surface = pygame.Surface((MAP_WIDTH * CELL_SIZE, MAP_HEIGHT * CELL_SIZE))

        # Loop through the map and draw it onto the surface
        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color = (0, 0, 0) if self.map[y, x] == 1 else (255, 255, 255)  # Black for walls, white for walkable
                pygame.draw.rect(map_surface, color, rect)

        # Save the surface as an image file
        pygame.image.save(map_surface, filename)
        print(f"Map saved as {filename}")


    def flood_fill(self, map_grid, start_pos):
        """Perform flood fill to find all connected walkable cells (white cells)."""
        room = []
        filled = np.zeros_like(map_grid, dtype=bool)
        to_fill = [start_pos]
        
        while to_fill:
            x, y = to_fill.pop()
            if not filled[y, x] and map_grid[y, x] == 0:
                filled[y, x] = True
                room.append((x, y))  # Collect the walkable positions in the room
                # Add adjacent walkable cells to be filled
                if x > 0: to_fill.append((x - 1, y))
                if x < MAP_WIDTH - 1: to_fill.append((x + 1, y))
                if y > 0: to_fill.append((x, y - 1))
                if y < MAP_HEIGHT - 1: to_fill.append((x, y + 1))
        
        return room

    def connect_rooms(self, map_grid):
        """Connect all isolated rooms by creating tunnels between them."""
        rooms = []
        visited = np.zeros_like(map_grid, dtype=bool)

        # Identify all isolated rooms
        for y in range(1, MAP_HEIGHT - 1):
            for x in range(1, MAP_WIDTH - 1):
                if map_grid[y, x] == 0 and not visited[y, x]:
                    room = self.flood_fill(map_grid, (x, y))
                    for pos in room:
                        visited[pos[1], pos[0]] = True  # Mark as visited
                    rooms.append(room)

        # Now connect all rooms
        if len(rooms) > 1:
            for i in range(len(rooms) - 1):
                room1 = rooms[i]
                closest_room, closest_point1, closest_point2 = self.find_closest_room(room1, rooms[i+1:])
                self.create_natural_tunnel(map_grid, closest_point1, closest_point2)

    def find_closest_room(self, room1, remaining_rooms):
        """Find the closest room to room1 and return the closest points between them."""
        closest_dist = float('inf')
        closest_point1, closest_point2 = None, None
        closest_room = None

        for room2 in remaining_rooms:
            for point1 in room1:
                for point2 in room2:
                    dist = abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_point1, closest_point2 = point1, point2
                        closest_room = room2

        return closest_room, closest_point1, closest_point2

    def create_natural_tunnel(self, map_grid, point1, point2):
        """Create a random winding tunnel between two points."""
        x1, y1 = point1
        x2, y2 = point2

        while x1 != x2 or y1 != y2:
            map_grid[y1, x1] = 0  # Carve out a walkable cell (white)

            # Randomly prioritize moving horizontally or vertically
            if random.random() < 0.5:  # Move in x direction
                if x1 != x2:
                    x1 += np.sign(x2 - x1)
            else:  # Move in y direction
                if y1 != y2:
                    y1 += np.sign(y2 - y1)

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

        # Check for collisions with walls
        for wall_rect in self.walls:
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

        # Update the camera's position based on the agent's new position
        self.update_camera()

        return self.get_observation(), 0, False, {}

    def reset(self):
        # Reset the agent position to a random walkable cell
        self.agent_pos = self.get_random_walkable_position()

        # Reset the camera position
        self.update_camera()

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
import pygame
import numpy as np
import random

from config import *

class MapManager():
    """Manager for the map."""

    def __init__(self):
        """Initialize MapManager and generate multiple maps."""
        self.maps = []  # List to store multiple maps
        self.walls_list = []  # List to store walls for each map
        self.special_cells_list = []  # List to store special cells for each map
        self.num_maps = NB_LEVELS
        self.current_map_index = 0  # Track which map is currently active
        self.num_goal = NB_GREEN_CELLS

        # Generate num_maps maps
        for _ in range(self.num_maps):
            self.generate_map()

    def generate_map(self):
        """Generate a single map and store it."""
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

        # Place the special cell (-1 value) on a random walkable spot
        self.place_special_cells(map_grid)

        # Create Rects for walls and special cells
        walls = []
        special_cells = []

        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                if map_grid[y, x] == 1:  # Wall
                    wall_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    walls.append(wall_rect)
                elif map_grid[y, x] == -1:  # Special green cell
                    special_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    special_cells.append(special_rect)

        # Store the map, walls, and special cells for this map
        self.maps.append(map_grid)
        self.walls_list.append(walls)
        self.special_cells_list.append(special_cells)

        self.set_map(len(self.maps) - 1)
        # Save each map as an image (optional for debugging)
        self.save_map_as_image(f"map_{len(self.maps)}.png")

    def set_map(self, index):
        """Set the current map to the one at the specified index."""
        if 0 <= index < self.num_maps:
            self.current_map_index = index
            self.map = self.maps[index]
            self.walls = self.walls_list[index]
            self.special_cells = self.special_cells_list[index]
        else:
            print(f"Map index {index} out of range.")

    def place_special_cells(self, map_grid):
        """Randomly place multiple special cells (-1) on the map."""
        walkable_cells = [(x, y) for y in range(1, MAP_HEIGHT - 1) for x in range(1, MAP_WIDTH - 1) if map_grid[y, x] == 0]
        if len(walkable_cells) >= self.num_goal:
            selected_cells = random.sample(walkable_cells, self.num_goal)  # Randomly select 10 unique cells
            for x, y in selected_cells:
                map_grid[y, x] = -1  # Assign -1 to mark the special green cells


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

    def save_map_as_image(self, filename="generated_map.png"):
        """Save the generated map as an image file."""
        map_surface = pygame.Surface((MAP_WIDTH * CELL_SIZE, MAP_HEIGHT * CELL_SIZE))
        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.map[y, x] == 1:
                    color = (0, 0, 0)  # Black for walls
                elif self.map[y, x] == -1:
                    color = (0, 255, 0)  # Green for the special cell
                else:
                    color = (255, 255, 255)  # White for walkable cells
                pygame.draw.rect(map_surface, color, rect)

        pygame.image.save(map_surface, "./media/pictures/" + filename)
        #print(f"Map saved as {filename}")
import pygame

from classes.CustomEnv import *
import pygame
from classes.CustomEnv import CustomEnv

from config import *

# Main function
if __name__ == "__main__":
    env = CustomEnv(render_mode="human")  # Set render mode to human for display
    obs = env.reset()
    running = True
    terminated = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Determine action based on key presses
        if pygame.key.get_pressed()[pygame.K_UP]:
            action = 1  # Action 1 when UP arrow is pressed
        else:
            action = 0  # Action 0 when no key is pressed

        # Perform the action in the environment
        state, reward, terminated, truncated, info = env.step(action)

        running = not terminated

        # Render the environment
        env.render()

    pygame.quit()

# # Main function
# if __name__ == "__main__":
#     train_agents()
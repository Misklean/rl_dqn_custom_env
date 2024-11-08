import pygame

from classes.CustomEnv import *
import pygame
from classes.CustomEnv import CustomEnv
from train_model import train_model

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

# # Main function
# if __name__ == "__main__":
#     train_model()
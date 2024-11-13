import pygame

from classes.CustomEnv import *
import pygame
from classes.CustomEnv import CustomEnv
from classes.DQN import train_multiple_agents

# # Main function
# if __name__ == "__main__":
#     env = CustomEnv()
#     obs = env.reset()
#     running = True
#     terminated = False

#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False

#         # Move the agent based on key presses
#         keys = pygame.key.get_pressed()
#         if keys[pygame.K_UP]:
#             state, reward, terminated, truncated, info = env.step(0)  # up
#         if keys[pygame.K_DOWN]:
#             state, reward, terminated, truncated, info = env.step(1)  # down
#         if keys[pygame.K_LEFT]:
#             state, reward, terminated, truncated, info = env.step(2)  # left
#         if keys[pygame.K_RIGHT]:
#             state, reward, terminated, truncated, info = env.step(3)  # right

#         running = not terminated

#         # Render the environment
#         env.render()

#     pygame.quit()

# Main function
if __name__ == "__main__":
    train_multiple_agents()
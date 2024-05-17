import math
from typing import Optional

import keyboard
import numpy as np
import pygame
import sys

from src_files.Environments.Abstract_Environment.Abstract_Environment import Abstract_Environment
from src_files.Environments.general_functions_provider import get_environment_class
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Models.Normal.Normal_model import Normal_model
from src_files.constants import CONSTANTS_DICT

# Constants
environment_name = CONSTANTS_DICT["environment"]["name"]
environment_kwargs = {
    **CONSTANTS_DICT["environment"]["universal_kwargs"],
    **CONSTANTS_DICT["environment"]["changeable_validation_kwargs_list"][0],
}
width, height = environment_kwargs["map_view"].shape
car_image_path = CONSTANTS_DICT["visualization"]["car_image_path"]
map_image_path = CONSTANTS_DICT["visualization"]["map_image_path"]
car_dimmensions = CONSTANTS_DICT["environment"]["universal_kwargs"]["car_dimensions"]



class Car:
    def __init__(self, environment: Abstract_Environment, model: Optional[Normal_model] = None):
        self.environment = environment
        self.model = model
        self.sprite = pygame.image.load(car_image_path).convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (car_dimmensions[1], car_dimmensions[0]))
        self.position = self.environment.get_car_position()
        self.angle = self.environment.get_car_angle()

    def reset(self):
        self.environment.p_reset()

    def getDrawPosition(self):
        rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)
        draw_position = (
            self.position[0] - rotated_sprite.get_width() / 2, self.position[1] - rotated_sprite.get_height() / 2)
        return rotated_sprite, draw_position

    def draw(self, screen):
        rotated_sprite, draw_position = self.getDrawPosition()
        screen.blit(rotated_sprite, draw_position)

    def step(self) -> bool:
        """
        Reacts to keyboard input
        :return: true if alive, false if dead
        """

        if self.model is not None:
            state = np.array(self.environment.p_get_state(), dtype=np.float32).reshape(1, -1)
            output = self.model.p_forward_pass(state)[0]
        else:
            output = np.zeros(3, dtype=np.float32)
            if keyboard.is_pressed('left'):
                output[1] = 1
            elif keyboard.is_pressed('right'):
                output[2] = 1

        self.environment.p_react(output)
        self.position = self.environment.get_car_position()
        self.angle = self.environment.get_car_angle()

        # Check for collision
        return self.environment.p_is_alive()


def run_basic_environment_visualization(model: Optional[Normal_model] = None):
    environment = get_environment_class(environment_name)(**environment_kwargs)

    clock = pygame.time.Clock()
    pygame.init()
    screen = pygame.display.set_mode((width, height))

    car = Car(environment, model)

    # Load the map
    game_map = pygame.image.load(map_image_path).convert()
    game_map = pygame.transform.scale(game_map, (width, height))

    is_alive = True

    while is_alive:
        clock.tick(150)
        # keys = pygame.key.get_pressed()
        is_alive = car.step()

        screen.fill((0, 0, 0))  # Clear the screen
        screen.blit(game_map, (0,0))  # Draw the map
        car.draw(screen)  # Draw the car on the screen

        pygame.display.flip()  # Update the display

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                #sys.exit()

        if not is_alive:
            pygame.quit()
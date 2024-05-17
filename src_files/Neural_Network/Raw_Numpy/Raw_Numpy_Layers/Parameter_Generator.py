from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np


class ParameterGenerator(ABC):
    @abstractmethod
    def generate_weights(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Generates weights
        :param shape: shape of weights
        :return: np.ndarray
        """
        pass

    @abstractmethod
    def generate_biases(self, shape: Tuple[int]) -> np.ndarray:
        """
        Generates biases
        :param shape: shape of biases
        :return: np.ndarray
        """
        pass


class Normal_Distribution_Generator(ParameterGenerator):
    def __init__(self, mean: float = 0.0, std: float = 0.05):
        """

        :param mean: default 0
        :param std: default 0.05
        """
        self.mean = mean
        self.std = std

    def generate_weights(self, shape: List[int]) -> np.ndarray:
        """
        Generates weights, normal distribution with mean and std from constructor
        :param shape:
        :return:
        """
        return np.random.normal(self.mean, self.std, shape).astype(np.float32)

    def generate_biases(self, shape: List[int]) -> np.ndarray:
        """
        Generates biases, zeros
        :param shape:
        :return:
        """
        return np.zeros(shape, dtype=np.float32)
        # return np.random.normal(self.mean, self.std, shape).astype(np.float32)


class Xavier_Distribution_Generator(ParameterGenerator):
    def generate_weights(self, shape: List[int]) -> np.ndarray:
        """
        Generates weights, xavier distribution for tanh and sigmoid
        :param shape: should be 2d, (input, output)
        :return:
        """
        return np.random.normal(0, np.sqrt(2.0 / (shape[0] + shape[1])), shape).astype(np.float32)

    def generate_biases(self, shape: List[int]) -> np.ndarray:
        """
        Generates biases, zeros
        :param shape:
        :return:
        """
        return np.zeros(shape, dtype=np.float32)
        # return np.random.normal(0, np.sqrt(2.0 / (shape[0])), shape).astype(np.float32)

class He_Distribution_Generator(ParameterGenerator):
    def generate_weights(self, shape: List[int]) -> np.ndarray:
        """
        Generates weights, he distribution, for relu
        :param shape: should be 2d, (input, output)
        :return: np.ndarray
        """
        return np.random.normal(0, np.sqrt(2.0 / (shape[0])), shape).astype(np.float32)

    def generate_biases(self, shape: List[int]) -> np.ndarray:
        """
        Generates biases, zeros
        :param shape:
        :return:
        """
        return np.zeros(shape, dtype=np.float32)
        # return np.random.normal(0, np.sqrt(2.0 / (shape[0])), shape).astype(np.float32)
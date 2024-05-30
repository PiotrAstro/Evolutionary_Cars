

import math
from threading import Lock
from typing import Any

import numpy as np

from src_files.Evolutionary_Algorithms._depracated_Immutable_Individual import Immutable_Individual
from src_files.Evolutionary_Algorithms.Mutation_Controllers.mutation_controllers_functions import \
    Abstract_Mutation_Controller


class Mut_Prob_Epochs(Abstract_Mutation_Controller):
    change_rate: float
    mutation_factors: np.ndarray
    initial_mut_fact_range: tuple[float, float]
    factors_of_prev_individuals: dict[Immutable_Individual, float]
    factors_of_new_individuals: dict[Immutable_Individual, float]
    factors_of_prev_parents: dict[Immutable_Individual, float]
    factors_of_new_parents: dict[Immutable_Individual, float]

    def __init__(self, initial_mut_fact_range: tuple[float, float], change_rate: float):
        self.initial_mut_fact_range = initial_mut_fact_range
        self.change_rate = change_rate
        self.factors_of_prev_individuals = {}
        self.factors_of_new_individuals = {}
        self.factors_of_prev_parents = {}
        self.factors_of_new_parents = {}

        mem_size = 10
        change_rate = (initial_mut_fact_range[1] / initial_mut_fact_range[0]) ** (1 / (mem_size - 1))
        self.mutation_factors = np.empty(mem_size)
        current_value = initial_mut_fact_range[0]
        for i in range(mem_size):
            self.mutation_factors[i] = current_value
            current_value *= change_rate
        # self.global_changes_tmp = np.zeros(mem_size)

    def mutate(self, child: Immutable_Individual, parent: Immutable_Individual):
        if parent not in self.factors_of_prev_parents:
            # self.factors_of_individuals[parent] = np.random.uniform(*self.initial_mut_fact_range)
            if parent not in self.factors_of_prev_individuals:
                self.factors_of_prev_individuals[parent] = np.random.choice(self.mutation_factors)
            self.factors_of_prev_parents[parent] = self.factors_of_prev_individuals[parent]
        self.factors_of_new_parents[parent] = self.factors_of_prev_parents[parent]
        mut_fact = self.factors_of_prev_parents[parent]
        change = np.random.normal(1, self.change_rate)
        if change > 1:
            change = (change - 1) / (1 - self.change_rate) + 1
        mut_fact *= change

        # if np.random.rand() < 0.1:
        #     mut_fact = np.random.choice(self.mutation_factors)

        mut_fact = max(self.initial_mut_fact_range[0], min(self.initial_mut_fact_range[1], mut_fact))
        self.factors_of_new_individuals[child] = mut_fact

        params = child.neural_network.get_parameters()
        self._permute(params, mut_fact)
        child.neural_network.set_parameters(params)

    def commit_iteration(self) -> None:
        self.factors_of_prev_individuals = self.factors_of_new_individuals
        self.factors_of_new_individuals = {}
        self.factors_of_prev_parents = self.factors_of_new_parents
        self.factors_of_new_parents = {}

        cups_values = self._sum_in_cups(self.mutation_factors, list(self.factors_of_prev_parents.values()))
        cups_probs = cups_values / np.sum(cups_values)
        print(cups_probs)
        # if len(self.factors_of_parents) > 1000:
        #     print("\n\n\nReducing Memory\n\n\n")
        #     self.factors_of_parents = {key: value for key, value in self.factors_of_parents.items() if np.random.rand() < 0.5}

    def _sum_in_cups(self, cups_values: np.ndarray, values: np.ndarray | list[float]) -> np.ndarray:
        """
        Sums values in cups
        """
        change_rate = (cups_values[-1] / cups_values[0]) ** (1 / (len(cups_values) - 1))
        cups = np.zeros(len(cups_values))
        for value in values:
            index = int(math.log(value / cups_values[0], change_rate))
            cups[index] += 1
        return cups


    def _permute(self, param: dict[str, Any] | np.ndarray, mut_factor: float) -> None:
        """
        Permutes parameters dictionary inplace
        """
        if isinstance(param, np.ndarray):
            param += np.random.normal(0, mut_factor, param.shape)
        else:
            for key, value in param.items():
                self._permute(value, mut_factor)

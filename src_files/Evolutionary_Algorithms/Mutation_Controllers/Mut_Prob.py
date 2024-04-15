import math
from threading import Lock
from typing import Any

import numpy as np

from src_files.Evolutionary_Algorithms.Mutation_Controllers.mutation_controllers_functions import \
    Abstract_Mutation_Controller


class Mut_Prob(Abstract_Mutation_Controller):
    mut_size: int
    lock: Lock
    learning_rate: float
    survival_rate: float
    mutation_factors: np.ndarray
    mutation_probs: np.ndarray
    mutation_indecies: np.ndarray
    initial_mut_fact_range: tuple[float, float]
    improved_over_parents: list[tuple[int, float, float]]  # mut index, parent fitness, child fitness

    def __init__(self, mem_size: int, initial_mut_fact_range: tuple[float, float], learning_rate: float, survival_rate: float):
        self.initial_mut_fact_range = initial_mut_fact_range
        self.mut_size = mem_size
        self.learning_rate = learning_rate
        self.survival_rate = survival_rate
        change_rate = (initial_mut_fact_range[1] / initial_mut_fact_range[0]) ** (1 / (mem_size - 1))
        self.mutation_factors = np.empty(mem_size)
        current_value = initial_mut_fact_range[0]
        for i in range(mem_size):
            self.mutation_factors[i] = current_value
            current_value *= change_rate

        self.mutation_indecies = np.arange(mem_size)
        self.mutation_probs = np.ones(mem_size) / mem_size
        self.improved_over_parents = []
        self.lock = Lock()
        # self.global_changes_tmp = np.zeros(mem_size)

    def mutate(self, params: dict[str, Any] | np.ndarray) -> int:
        mut_fact_new = np.random.choice(self.mutation_indecies, p=self.mutation_probs)

        self._permute(params, self.mutation_factors[mut_fact_new])
        return mut_fact_new

    def mutation_better_than_parent(self, id: int, parent_fitness: float, child_fitness: float) -> None:
        self.lock.acquire()
        self.improved_over_parents.append((id, parent_fitness, child_fitness))
        self.lock.release()

    def commit_iteration(self, fitnesses: np.ndarray) -> None:
        new_probs = np.ones(self.mut_size) / self.mut_size

        if self.improved_over_parents:
            threshold = np.quantile(fitnesses, 0.95)
            new_probs_tmp = np.zeros(self.mut_size)
            for index, parent_fitness, child_fitness in self.improved_over_parents:
                candidate = child_fitness - parent_fitness
                if child_fitness > threshold:
                    new_probs_tmp[index] += candidate
                # if candidate > new_probs_tmp[index]:
                #     new_probs_tmp[index] = candidate
            # new_probs_tmp *= (1 - self.mutation_probs)
            new_probs_tmp /= np.sum(new_probs_tmp)
            # self.global_changes_tmp += new_probs_tmp
            # new_probs_tmp = self.global_changes_tmp / np.sum(self.global_changes_tmp)
            new_probs = self.survival_rate * new_probs + (1 - self.survival_rate) * new_probs_tmp

            self.improved_over_parents = []
        self.mutation_probs = (1 - self.learning_rate) * self.mutation_probs + self.learning_rate * new_probs
        self.mutation_probs /= np.sum(self.mutation_probs)
        self.current_mutations = []


    def _permute(self, param: dict[str, Any] | np.ndarray, mut_factor: float) -> None:
        """
        Permutes parameters dictionary inplace
        """
        if isinstance(param, np.ndarray):
            param += np.random.normal(0, mut_factor, param.shape)
        else:
            for key, value in param.items():
                self._permute(value, mut_factor)

    def __str__(self) -> str:
        number_to_show = 10
        stride = self.mut_size // number_to_show
        mut_probs = np.convolve(self.mutation_probs, np.ones(stride), mode="valid")[::stride]
        # global_changes = np.convolve(self.global_changes_tmp, np.ones(stride), mode="valid")[::stride]
        return " ".join([f"{x:.3f}" for x in mut_probs])  # + "\n" + " ".join([f"{x:.3f}" for x in global_changes])

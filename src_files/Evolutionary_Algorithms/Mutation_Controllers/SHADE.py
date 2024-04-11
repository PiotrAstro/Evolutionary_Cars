from threading import Lock
from typing import Any

import numpy as np

from src_files.Evolutionary_Algorithms.Mutation_Controllers.mutation_controllers_functions import \
    Abstract_Mutation_Controller


class SHADE(Abstract_Mutation_Controller):
    mem_size: int
    lock: Lock
    mut_change_sigma: float
    mutation_factors: np.ndarray
    k: int
    current_mutations: list[float]  # mutation factor after mutation
    improved_over_parents: list[tuple[float, float, float]]  # new mutation factor, parent fitness, child fitness
    initial_mut_fact_range: tuple[float, float]

    def __init__(self, mem_size: int, initial_mut_fact_range: tuple[float, float], mut_change_sigma: float):
        self.mem_size = mem_size
        self.mut_change_sigma = mut_change_sigma
        self.initial_mut_fact_range = initial_mut_fact_range
        self.mutation_factors = np.linspace(initial_mut_fact_range[0], initial_mut_fact_range[1], mem_size)
        self.mutation_factors = np.random.permutation(self.mutation_factors)
        self.k = 0
        self.current_mutations = []
        self.improved_over_parents = []
        self.lock = Lock()

    def mutate(self, params: dict[str, Any]) -> int:
        mut_fact_org = np.random.choice(self.mutation_factors)
        rand = np.random.normal(0, self.mut_change_sigma)
        rand *= (1 - self.mut_change_sigma) if rand < 0 else 1
        rand += 1
        mut_fact_new = mut_fact_org * rand
        mut_fact_new = max(min(mut_fact_new, self.initial_mut_fact_range[1]), self.initial_mut_fact_range[0])

        self.lock.acquire()
        self.current_mutations.append(mut_fact_new)
        id_mutation = len(self.current_mutations) - 1
        self.lock.release()

        self._permute_params_dict(params, mut_fact_new)
        return id_mutation

    def mutation_better_than_parent(self, id: int, parent_fitness: float, child_fitness: float) -> None:
        self.improved_over_parents.append((self.current_mutations[id], parent_fitness, child_fitness))

    def commit_iteration(self) -> None:
        if self.improved_over_parents:
            tmp_array = np.array(self.improved_over_parents)
            new_mutations = tmp_array[:, 0]
            parent_fitness = tmp_array[:, 1]
            child_fitness = tmp_array[:, 2]

            improvement = child_fitness - parent_fitness
            weights = improvement / np.sum(improvement)
            new_value = np.sum(new_mutations * weights)
            self.mutation_factors[self.k] = new_value

            self.improved_over_parents = []
            self.k = (self.k + 1) % self.mem_size
        self.current_mutations = []


    def _permute_params_dict(self, param_dict: dict[str, Any], mut_factor: float) -> None:
        """
        Permutes parameters dictionary inplace
        """
        for key, value in param_dict.items():
            if isinstance(value, dict):
                self._permute_params_dict(value, mut_factor)
            elif isinstance(value, np.ndarray):
                value += np.random.normal(0, mut_factor, value.shape)
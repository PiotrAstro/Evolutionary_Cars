from threading import Lock
from typing import Any

import numpy as np

from src_files.Evolutionary_Algorithms.Mutation_Controllers.mutation_controllers_functions import \
    Abstract_Mutation_Controller

class SHADE_single(Abstract_Mutation_Controller):
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

    def mutate(self, params: dict[str, Any] | np.ndarray) -> int:
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

        self._permute(params, mut_fact_new)
        return id_mutation

    def mutation_better_than_parent(self, id: int, parent_fitness: float, child_fitness: float) -> None:
        self.lock.acquire()
        if id >= len(self.current_mutations):
            raise ValueError("Mutation id out of range")
        self.improved_over_parents.append((self.current_mutations[id], parent_fitness, child_fitness))
        self.lock.release()

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
        mut_fact = list(sorted(self.mutation_factors))[::5]
        return " ".join([f"{x:.3f}" for x in mut_fact])


class SHADE_multiple(Abstract_Mutation_Controller):
    dict_SHADEs: dict[str, Any]
    list_SHADEs: list[SHADE_single]
    singular_args: tuple[Any, ...]
    singular_kwargs: dict[str, Any]
    lock: Lock

    def __init__(self, *args, **kwargs):
        self.singular_args = args
        self.singular_kwargs = kwargs
        self.lock = Lock()
        self.dict_SHADEs = {}
        self.list_SHADEs = []

    def mutate(self, params: dict[str, Any]) -> int:
        return self._permute(params, self.dict_SHADEs)

    def _permute(self, param: dict[str, Any] | np.ndarray, SHADE_dict: dict[str, Any] | SHADE_single) -> int:
        """
        Permutes parameters dictionary inplace
        returns index of mutation
        """
        if isinstance(param, np.ndarray):
            return SHADE_dict.mutate(param)
        else:
            returned_id = -1
            for key, value in param.items():
                self.lock.acquire()
                if key not in SHADE_dict:
                    if isinstance(value, dict):
                        SHADE_dict[key] = {}
                    else:
                        new = SHADE_single(*self.singular_args, **self.singular_kwargs)
                        SHADE_dict[key] = new
                        self.list_SHADEs.append(new)
                self.lock.release()
                returned_id = self._permute(value, SHADE_dict[key])
            return returned_id

    def mutation_better_than_parent(self, id: int, parent_fitness: float, child_fitness: float) -> None:
        for shade in self.list_SHADEs:
            shade.mutation_better_than_parent(id, parent_fitness, child_fitness)

    def commit_iteration(self) -> None:
        for shade in self.list_SHADEs:
            shade.commit_iteration()

    def __str__(self) -> str:

        return "\n".join([shade.__str__() for shade in self.list_SHADEs])


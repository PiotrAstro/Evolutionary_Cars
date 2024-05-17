from typing import Any

import numpy as np

from src_files.Evolutionary_Algorithms.Immutable_Individual import Immutable_Individual
from src_files.Evolutionary_Algorithms.Mutation_Controllers.mutation_controllers_functions import \
    Abstract_Mutation_Controller


class Mut_One(Abstract_Mutation_Controller):
    mutation_factor: float
    use_children: bool
    moving_std: float

    def __init__(self, mutation_factor: float, use_children: bool) -> None:
        self.mutation_factor = mutation_factor
        self.use_children = use_children
        self.moving_std = -1

    def mutate(self, child: Immutable_Individual, parent: Immutable_Individual) -> None:
        params = child.neural_network.get_parameters()

        if self.use_children:
            children_params = [params] + [child.neural_network.get_parameters() for child in parent.children]
            fitnesses = np.array([parent.get_fitness()] + [child.get_fitness() for child in parent.children])
            std = np.std(fitnesses)
            if std == 0:
                std = 1
            fitnesses = (fitnesses - np.mean(fitnesses)) / std
            self._permute(params, children_params, fitnesses, self.mutation_factor)
        else:
            self._permute(params, [], None, self.mutation_factor)
        child.neural_network.set_parameters(params)


    def commit_iteration(self, fitnesses: np.ndarray) -> None:
        pass
        # new_std = np.std(fitnesses)
        #
        # moving_factor = 0.05
        # mutation_factor = 0.95
        #
        # if self.moving_std < 0:
        #     self.moving_std = new_std
        # else:
        #     self.moving_std = moving_factor * new_std + (1 - moving_factor) * self.moving_std
        #     self.mutation_factor *= mutation_factor ** np.clip((new_std - self.moving_std) / self.moving_std, -10, 10)
        #     if self.mutation_factor < 0.01:
        #         self.mutation_factor = 0.01
        #         self.moving_std = new_std
        #     elif self.mutation_factor > 0.2:
        #         self.mutation_factor = 0.2
        #         self.moving_std = new_std
        # print(f"Mutation factor: {self.mutation_factor}, moving std: {self.moving_std}, new std: {new_std}")

    def _permute(self, param: dict[str, Any] | np.ndarray, child_params: list[dict[str, Any] | np.ndarray], fitnesses: np.ndarray | None, mut_factor: float) -> None:
        """
        Permutes parameters dictionary inplace
        """
        if isinstance(param, np.ndarray):
            param += np.random.normal(0, mut_factor, param.shape)
            # param += np.random.normal(0, max(np.std(param), 0.001) * mut_factor, param.shape)
            # param *= mut_factor ** np.random.normal(0, 1, param.shape)
            if self.use_children:
                change = np.zeros(param.shape)
                for child, fitness in zip(child_params, fitnesses):
                    change += (child - param) * fitness * mut_factor
                param += change
        else:
            for key, value in param.items():
                self._permute(value, [child[key] for child in child_params], fitnesses, mut_factor)

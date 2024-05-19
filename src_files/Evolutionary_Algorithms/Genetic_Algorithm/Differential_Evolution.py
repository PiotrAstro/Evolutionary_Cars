import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Type, List

import numpy as np
import pandas as pd

from src_files.Environments.Abstract_Environment.Abstract_Environment import Abstract_Environment
from src_files.Environments.Abstract_Environment.Abstract_Environment_Iterator import Abstract_Environment_Iterator
from src_files.Environments.general_functions_provider import get_environment_class
from src_files.Environments_Visualization.Basic_Environment_Visualization import run_basic_environment_visualization
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Models.Normal.Normal_model import Normal_model


class Differential_Evolution:
    """
    Differential Evolution - https://en.wikipedia.org/wiki/Differential_evolution
    """
    def __init__(self, constants_dict: Dict[str, Any]) -> None:
        """
        Initializes Differential Evolution - https://en.wikipedia.org/wiki/Differential_evolution
        :param constants_dict:
        :return:
        """
        seed = int(time.time() * 10000) % 2**32
        np.random.seed(seed)

        # things taken from constants_dict:
        self.population_size = constants_dict["Differential_Evolution"]["population"]
        self.epochs = constants_dict["Differential_Evolution"]["epochs"]
        self.cross_prob = constants_dict["Differential_Evolution"]["cross_prob"]
        self.diff_weight = constants_dict["Differential_Evolution"]["diff_weight"]
        self.save_logs_every_n_epochs = constants_dict["Differential_Evolution"]["save_logs_every_n_epochs"]
        self.max_evaluations = constants_dict["Differential_Evolution"]["max_evaluations"]
        self.max_threads = os.cpu_count() if constants_dict["Differential_Evolution"]["max_threads"] <= 0 else constants_dict["Differential_Evolution"]["max_threads"]
        base_log_dir = constants_dict["Differential_Evolution"]["logs_path"]

        self.training_environments_kwargs = [
            {
                **constants_dict["environment"]["universal_kwargs"],
                **testing_kwargs,
            } for testing_kwargs in constants_dict["environment"]["changeable_training_kwargs_list"]
        ]
        self.validation_environments_kwargs = [
            {
                **constants_dict["environment"]["universal_kwargs"],
                **validation_kwargs,
            } for validation_kwargs in constants_dict["environment"]["changeable_validation_kwargs_list"]
        ]
        self.environment_class = get_environment_class(constants_dict["environment"]["name"])
        assert self.environment_class is not None
        # end of things taken from constants_dict

        # file handling
        self.log_directory = base_log_dir + "/" + "DiffEv" + str(int(time.time())) + "/"
        # os.makedirs(self.log_directory, exist_ok=True)

        #self.logger = Timestamp_Logger(file_path=self.log_directory + "log.txt", log_mode='w', log_moment='a', separator='\t')

        self.neural_network_kwargs = constants_dict["neural_network"]
        self.population = [
            Individual(self.neural_network_kwargs, self.environment_class, self.training_environments_kwargs)
            for _ in range(self.population_size)
        ]

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(individual.get_fitness) for individual in self.population]

            results = [future.result() for future in futures]

        self.best_individual = max(self.population, key=lambda individual: individual.get_fitness()).copy()


    def run(self) -> pd.DataFrame:
        """
        Runs evolutionary algorithm
        :return:
        """
        log_list = []
        for generation in range(self.epochs):
            print(f"Generation {generation}")

            time_start = time.perf_counter()
            # multi-threading
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                futures = []
                for i in range(self.population_size):
                    base_id, id1, id2 = np.random.choice(self.population_size, 3, replace=False)
                    base_params = self.best_individual.neural_network.get_parameters()  # self.population[base_id].neural_network.get_parameters()
                    id1_params = self.population[id1].neural_network.get_parameters()
                    id2_params = self.population[id2].neural_network.get_parameters()
                    futures.append(executor.submit(self.population[i].differential_evolution_one_epoch, base_params, id1_params, id2_params, self.cross_prob, self.diff_weight))

                results = [future.result() for future in futures]
            # for i in range(self.crosses_per_poch):
            #     self.__perform_cross(indecies_randomized[i * 2], indecies_randomized[i * 2 + 1])
            # end of multi-threading
            time_end = time.perf_counter()
            print(f"Time: {time_end - time_start}, mean time: {(time_end - time_start) / self.population_size / 2}, mean time using one thread: {(time_end - time_start) / self.population_size / 2 * self.max_threads}")

            for individual in self.population:
                if individual.get_fitness() > self.best_individual.get_fitness():
                    self.best_individual = individual.copy()

            fitnesses = np.array([individual.get_fitness() for individual in self.population])
            quantile = [0.25, 0.5, 0.75, 0.9, 0.99]
            quantile_results = np.quantile(fitnesses, quantile)
            quantile_text = ", ".join([f"{quantile}: {quantile_results[i]}" for i, quantile in enumerate(quantile)])
            print(f"Mean fitness: {fitnesses.mean()}, best fitness: {self.best_individual.get_fitness()}")
            print(f"Quantiles: {quantile_text}\n\n")

            evaluations = (generation + 2) * self.population_size

            if generation % self.save_logs_every_n_epochs == 0:
                # model = self.best_individual.neural_network
                # run_basic_environment_visualization(model)
                log_list.append({
                    "generation": generation,
                    "mean_fitness": fitnesses.mean(),
                    "best_fitness": self.best_individual.get_fitness(),
                    "evaluations": evaluations,
                })

            if evaluations > self.max_evaluations:
                break

        return pd.DataFrame(log_list)


class Individual:
    def __init__(self,
                 neural_network_params: Dict[str, Any],
                 environment_class: Type[Abstract_Environment],
                 environments_list_kwargs: List[Dict[str, Any]]) -> None:
        """
        Initializes Individual
        :param neural_network_params: parameters for neural network
        :param environment_class: class of environment
        :param environments_list_kwargs: list of kwargs for environments
        :param mutation_factor: float
        :param parent: parent of individual
        :param use_safe_mutation: bool - if True, then uses scaled mutation
        :param L1_factor: float - shrinks parameters by constant factor
        :param L2_factor: float - shrinks parameters using also parameters value
        """
        self.neural_network = Normal_model(**neural_network_params)

        self.neural_network_params = neural_network_params
        self.environment_class = environment_class
        self.environments_kwargs = environments_list_kwargs

        self.environments = [environment_class(**kwargs) for kwargs in environments_list_kwargs]
        self.environment_iterator = Abstract_Environment_Iterator(self.environments)

        self.fitness = 0.0
        self.is_fitness_calculated = False

    def get_fitness(self) -> float:
        if not self.is_fitness_calculated:
            self.fitness = self.environment_iterator.get_results(self.neural_network)
            self.is_fitness_calculated = True
        return self.fitness

    def copy(self) -> 'Individual':
        new_individual = Individual(self.neural_network_params,
                                    self.environment_class,
                                    self.environments_kwargs)
        new_individual.neural_network.set_parameters(self.neural_network.get_parameters())
        new_individual.fitness = self.fitness
        new_individual.is_fitness_calculated = self.is_fitness_calculated
        return new_individual

    def mutate(self, factor: float):
        params = self.neural_network.get_parameters()
        self._permute_params(params, factor)
        self.neural_network.set_parameters(params)
        self.is_fitness_calculated = False

    def _permute_params(self, params: dict[str, Any], factor: float) -> None:
        for key in params:
            if isinstance(params[key], np.ndarray):
                params[key] += np.random.normal(0, factor, params[key].shape)
            elif isinstance(params[key], dict):
                self._permute_params(params[key], factor)

    def differential_evolution_one_epoch(self,
                                         base_params: Dict[str, Any],
                                         individual_1_params: Dict[str, Any],
                                         individual_2_params: Dict[str, Any],
                                         cross_prob: float,
                                         diff_weight: float) -> bool:
        """
        Performs one step of differential evolution, modifies self inplace, if modification is better than base individual, returns True, otherwise False
        :param base_params:
        :param individual_1_params:
        :param individual_2_params:
        :param cross_prob:
        :param diff_weight:
        :return: True if change accepted, False otherwise
        """
        original_params = self.neural_network.get_parameters()
        original_fitness = self.get_fitness()

        new_params = self.neural_network.get_parameters()
        self._diff_evolution_iteration(new_params, base_params, individual_1_params, individual_2_params, cross_prob, diff_weight)

        self.neural_network.set_parameters(new_params)
        self.is_fitness_calculated = False

        if self.get_fitness() > original_fitness:
            return True
        else:
            self.neural_network.set_parameters(original_params)
            self.is_fitness_calculated = True
            self.fitness = original_fitness
            return False

    def _diff_evolution_iteration(self,
                                   self_params: Dict[str, Any],
                                    base_params: Dict[str, Any],
                                    individual_1_params: Dict[str, Any],
                                    individual_2_params: Dict[str, Any],
                                    cross_prob: float,
                                    diff_weight: float) -> None:
        """
        Performs Differential Evolution on np.array, iterates fthrwo params
        :param self_params:
        :param base_params:
        :param individual_1_params:
        :param individual_2_params:
        :param cross_prob:
        :param diff_weight:
        :return:
        """
        for key, value in self_params.items():
            if isinstance(value, dict):
                self._diff_evolution_iteration(value, base_params[key], individual_1_params[key], individual_2_params[key], cross_prob, diff_weight)
            elif isinstance(value, np.ndarray):
                cross_mask = np.random.rand(*value.shape) < cross_prob
                self_params[key][cross_mask] = base_params[key][cross_mask] + diff_weight * (individual_1_params[key][cross_mask] - individual_2_params[key][cross_mask])


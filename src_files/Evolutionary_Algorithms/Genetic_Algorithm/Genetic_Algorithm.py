import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Type

import numpy as np
import pandas as pd

from src_files.Environments.Abstract_Environment.Abstract_Environment import Abstract_Environment
from src_files.Environments.Abstract_Environment.Abstract_Environment_Iterator import Abstract_Environment_Iterator
from src_files.Environments.general_functions_provider import get_environment_class
from src_files.Environments_Visualization.Basic_Environment_Visualization import run_basic_environment_visualization
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Models.Normal.Normal_model import Normal_model


class Genetic_Algorithm:
    """
    Genetic Algorithm - sees neural network as set of genes, cross and mutate them
    """
    def __init__(self, constants_dict: Dict[str, Any]) -> None:
        """
        Initializes Genetic ALgorithm - cross and mutate neural networks
        :param constants_dict:
        :return:
        """
        seed = int(time.time() * 10000) % 2**32
        np.random.seed(seed)

        # things taken from constants_dict:
        self.population_size = constants_dict["Genetic_Algorithm"]["population"]
        self.epochs = constants_dict["Genetic_Algorithm"]["epochs"]
        self.mutation_factor = constants_dict["Genetic_Algorithm"]["mutation_factor"]
        self.crosses_per_epoch = constants_dict["Genetic_Algorithm"]["crosses_per_epoch"]
        # self.new_individual_every_n_epochs = constants_dict["Genetic_Algorithm"]["new_individual_every_n_epochs"]
        self.save_logs_every_n_epochs = constants_dict["Genetic_Algorithm"]["save_logs_every_n_epochs"]
        self.max_evaluations = constants_dict["Genetic_Algorithm"]["max_evaluations"]
        self.max_threads = os.cpu_count() if constants_dict["Genetic_Algorithm"]["max_threads"] <= 0 else constants_dict["Genetic_Algorithm"]["max_threads"]
        base_log_dir = constants_dict["Genetic_Algorithm"]["logs_path"]

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
        self.log_directory = base_log_dir + "/" + "GeAl" + str(int(time.time())) + "/"
        # os.makedirs(self.log_directory, exist_ok=True)

        #self.logger = Timestamp_Logger(file_path=self.log_directory + "log.txt", log_mode='w', log_moment='a', separator='\t')

        self.neural_network_kwargs = constants_dict["neural_network"]
        self.population = [
            Individual(self.neural_network_kwargs, self.environment_class, self.training_environments_kwargs)
            for _ in range(self.population_size)
        ]
        self.best_individual = self.population[0].copy()


    def run(self) -> pd.DataFrame:
        """
        Runs evolutionary algorithm
        :return:
        """
        log_list = []

        for generation in range(self.epochs):
            print(f"Generation {generation}")

            # if generation % self.new_individual_every_n_epochs == 0:
            #     self.population[random.randint(0, self.population_size - 1)] = Individual(self.neural_network_kwargs, self.environment_class, self.training_environments_kwargs)

            indecies_randomized = np.random.permutation(self.population_size)

            time_start = time.perf_counter()
            # multi-threading
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                futures = [
                    executor.submit(self._perform_cross, indecies_randomized[i * 2], indecies_randomized[i * 2 + 1])
                    for i in range(self.crosses_per_epoch)
                ]
                results = [future.result() for future in futures]
            # for i in range(self.crosses_per_epoch):
            #     self.__perform_cross(indecies_randomized[i * 2], indecies_randomized[i * 2 + 1])
            # end of multi-threading
            time_end = time.perf_counter()
            print(f"Time: {time_end - time_start}, mean time: {(time_end - time_start) / self.crosses_per_epoch / 2}, mean time using one thread: {(time_end - time_start) / self.crosses_per_epoch / 2 * self.max_threads}")

            for individual in self.population:
                if individual.get_fitness() > self.best_individual.get_fitness():
                    self.best_individual = individual.copy()

            fitnesses = np.array([individual.get_fitness() for individual in self.population])
            quantile = [0.25, 0.5, 0.75, 0.9, 0.99]
            quantile_results = np.quantile(fitnesses, quantile)
            quantile_text = ", ".join([f"{quantile}: {quantile_results[i]}" for i, quantile in enumerate(quantile)])
            print(f"Mean fitness: {fitnesses.mean()}, best fitness: {self.best_individual.get_fitness()}")
            print(f"Quantiles: {quantile_text}\n\n")

            evaluations = self.population_size + (1 + generation) * self.crosses_per_epoch * 2

            if generation % self.save_logs_every_n_epochs == 0:
                # model = self.best_individual.neural_network
                # run_basic_environment_visualization(model)
                log_list.append({
                    "generation": generation,
                    "evaluations": evaluations,
                    "mean_fitness": fitnesses.mean(),
                    "best_fitness": self.best_individual.get_fitness(),
                })

            if evaluations >= self.max_evaluations:
                break

        return pd.DataFrame(log_list)


    def _perform_cross(self, parent_index_1: int, parent_index_2: int) -> None:
        """
        Performs crosses, changes population in place
        :param parent_index_1: index of first parent
        :param parent_index_2: index of second parent
        """
        new_individual_1, new_individual_2 = self.population[parent_index_1].generate_crossed_scattered_individuals(self.population[parent_index_2])
        new_individual_1.mutate(self.mutation_factor)
        new_individual_2.mutate(self.mutation_factor)

        better_individual = new_individual_1 if new_individual_1.get_fitness() > new_individual_2.get_fitness() else new_individual_2
        worse_parent_index = parent_index_1 if self.population[parent_index_1].get_fitness() < self.population[parent_index_2].get_fitness() else parent_index_2
        self.population[worse_parent_index] = better_individual


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

    def generate_crossed_scattered_individuals(self, other: 'Individual') -> tuple['Individual', 'Individual']:
        """
        Generates two new individuals, which are scattered cross of two individuals
        :param other: other individual
        :return: two new individuals
        """
        self_policy_params = self.neural_network.get_parameters()
        other_policy_params = other.neural_network.get_parameters()
        self._cross_scattered_policy(self_policy_params, other_policy_params)
        new_individual1 = Individual(self.neural_network_params, self.environment_class, self.environments_kwargs)
        new_individual1.neural_network.set_parameters(self_policy_params)
        new_individual2 = Individual(self.neural_network_params, self.environment_class, self.environments_kwargs)
        new_individual2.neural_network.set_parameters(other_policy_params)
        return new_individual1, new_individual2

    def _cross_scattered_policy(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> None:
        """
        changes params_to_change so that on random places it swapes params1 and params2
        :param params1: first policy part of params
        :param params2: second policy part of params
        """
        for key, value in params1.items():
            if isinstance(value, dict):
                self._cross_scattered_policy(value, params2[key])
            elif isinstance(value, np.ndarray):
                mask = np.random.randint(0, 2, value.shape, dtype=np.bool_)
                params_2_mask_copy = np.array(params2[key][mask])
                params2[key][mask] = value[mask]
                params1[key][mask] = params_2_mask_copy


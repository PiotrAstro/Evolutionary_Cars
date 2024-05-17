import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Tuple, List, Mapping, Type

import numpy as np
import pandas as pd

from src_files.Environments.Abstract_Environment.Abstract_Environment import Abstract_Environment
from src_files.Environments.Abstract_Environment.Abstract_Environment_Iterator import Abstract_Environment_Iterator
from src_files.Environments.general_functions_provider import get_environment_class
from src_files.Environments_Visualization.Basic_Environment_Visualization import run_basic_environment_visualization
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Models.Normal.Normal_model import Normal_model


class Evolutionary_Strategy:
    """
    Evolutionary strategy calculating gradient per whole trajectory
    based on:
    Evolution Strategies as a Scalable Alternative to Reinforcement Learning
    """
    def __init__(self, constants_dict: Dict[str, Any]) -> None:
        """
        Initializes mutation only evolutionary algorithm
        :param constants_dict:
        :return:
        """
        seed = int(time.time())
        np.random.seed(seed)

        # things taken from constants_dict:
        self.permutations = constants_dict["Evolutionary_Strategy"]["permutations"]
        self.epochs = constants_dict["Evolutionary_Strategy"]["epochs"]
        self.sigma_change = constants_dict["Evolutionary_Strategy"]["sigma_change"]
        self.learning_rate = constants_dict["Evolutionary_Strategy"]["learning_rate"]
        self.stats_every_n_epochs = constants_dict["Evolutionary_Strategy"]["save_logs_every_n_epochs"]
        self.max_evaluations = constants_dict["Evolutionary_Strategy"]["max_evaluations"]
        self.max_threads = os.cpu_count() if constants_dict["Evolutionary_Strategy"]["max_threads"] <= 0 else constants_dict["Evolutionary_Strategy"]["max_threads"]
        base_log_dir = constants_dict["Evolutionary_Strategy"]["logs_path"]

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
        self.log_directory = base_log_dir + "/" + "EvSt" + str(int(time.time())) + "/"
        # os.makedirs(self.log_directory, exist_ok=True)

        # self.logger = Timestamp_Logger(file_path=self.log_directory + "log.txt", log_mode='w', log_moment='a', separator='\t')

        self.neural_network_kwargs = constants_dict["neural_network"]
        self.individual = Individual(self.neural_network_kwargs, self.environment_class, self.training_environments_kwargs)




    def run(self) -> pd.DataFrame:
        """
        Runs evolutionary algorithm
        :return:
        """
        log_list = []

        for generation in range(self.epochs):
            print(f"Generation {generation}")

            time_start = time.perf_counter()
            fitnesses = self.individual.evolutionary_strategy_one_epoch(self.permutations, self.sigma_change, self.learning_rate, self.max_threads)
            time_end = time.perf_counter()

            print(f"Time: {time_end - time_start}, mean time: {(time_end - time_start) / self.permutations}, mean time using one thread: {(time_end - time_start) / self.permutations * self.max_threads}")

            quantile = [0.25, 0.5, 0.75, 0.9, 0.99]
            quantile_results = np.quantile(fitnesses, quantile)
            quantile_text = ", ".join([f"{quantile}: {quantile_results[i]}" for i, quantile in enumerate(quantile)])
            print(f"Mean fitness: {fitnesses.mean()}, main fitness: {self.individual.get_fitness()}")
            print(f"Quantiles: {quantile_text}\n\n")


            evaluations = 1 + (self.permutations + 1) * (1 + generation)
            if generation % self.stats_every_n_epochs == 0:
                # model = self.individual.neural_network
                # run_basic_environment_visualization(model)
                log_list.append(
                    {
                        "generation": generation,
                        "main_fitness": self.individual.get_fitness(),
                        "best_fitness": max(fitnesses),
                        "evaluations": evaluations,
                    }
                )

            if evaluations >= self.max_evaluations:
                break

            # print(self.mutation_controller)
        log_data_frame = pd.DataFrame(log_list)

        return log_data_frame


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

    def copy_mutate_and_evaluate(self, factor: float) -> 'Individual':
        """
        Copies, mutates and evaluates individual
        :param mutation_factor:
        :param mutation_threshold: None or float - if not None, then it uses scaled mutation
        :return:
        """
        new_individual = self.copy()

        new_individual.mutate(factor)

        new_individual.get_fitness()

        return new_individual

    def evolutionary_strategy_one_epoch(self, number_of_individuals: int, sigma_change: float, alpha_learning_rate: float, num_of_processes: int) -> np.ndarray:
        """
        It performs one step of evolutionary strategy, modifies self inplace
        :param number_of_individuals:
        :param sigma_change:
        :param alpha_learning_rate:
        :param num_of_processes:
        :return: fitnesses of all mutated individuals
        """

        # multi-threading
        with ThreadPoolExecutor(max_workers=num_of_processes) as executor:
            futures = [
                executor.submit(self.copy_mutate_and_evaluate, sigma_change)
                for _ in range(number_of_individuals)
            ]
            individuals_mutated = [future.result() for future in futures]
        # end of multi-threading

        fitnesses = np.array([individual.get_fitness() for individual in individuals_mutated], dtype=float)
        fitnesses_normalized = (fitnesses - np.mean(fitnesses)) / np.std(fitnesses)
        multiply_factor = alpha_learning_rate / (number_of_individuals * sigma_change)
        change_amount = fitnesses_normalized * multiply_factor
        other_policies_params = [individual.neural_network.get_parameters() for individual in individuals_mutated]
        self_policy_params = self.neural_network.get_parameters()
        self._actualise_policy_dict_params(self_policy_params, other_policies_params, change_amount, [])
        self.neural_network.set_parameters(self_policy_params)

        self.is_fitness_calculated = False

        return fitnesses

    def _actualise_policy_dict_params(self, main_policy: Dict[str, Any], other_policies: List[Dict[str, Any]], change_amount: np.ndarray, key_list: List[str]) -> None:
        """
        Actualises policy dictionary parameters
        :param main_policy: current main policy part of params
        :param other_policies: list of other policies
        :param change_amount: fitnesses * multiply_factor
        :param key_list: list of keys to get to the parameter
        :return:
        """
        for key, value in main_policy.items():
            key_list.append(key)
            if isinstance(value, dict):
                self._actualise_policy_dict_params(value, other_policies, change_amount, key_list)
            elif isinstance(value, np.ndarray):
                param_array_double = value.astype(np.float64)
                for i in range(len(change_amount)):
                    param_array_double += change_amount[i] * self._get_element_from_dict(other_policies[i], key_list)
                main_policy[key] = param_array_double.astype(np.float32)
            key_list.pop()

    def _get_element_from_dict(self, dictionary: Dict[str, Any], key_list: List[str]) -> Any:
        """
        Gets element from dictionary by key list
        :param dictionary:
        :param key_list:
        :return:
        """
        for key in key_list:
            dictionary = dictionary[key]
        return dictionary

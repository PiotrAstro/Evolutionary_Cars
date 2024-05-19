from typing import Any, List, Dict, Type
import numpy as np
from src_files.Environments.Abstract_Environment.Abstract_Environment import Abstract_Environment
from src_files.Environments.Abstract_Environment.Abstract_Environment_Iterator import Abstract_Environment_Iterator
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Models.Normal.Normal_model import Normal_model

import os
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from src_files.Environments.general_functions_provider import get_environment_class


class GESMR:
    """
    Evolutionary Mutate Population - mutates population, then sorts new and old population and takes best half, then mutates again
    """
    def __init__(self, constants_dict: Dict[str, Any]) -> None:
        """
        Initializes mutation only evolutionary algorithm
        :param constants_dict:
        :return:
        """
        seed = int(time.time() * 10000) % 2**32
        np.random.seed(seed)

        # things taken from constants_dict:
        gesmr_dict = constants_dict["GESMR"]
        self.population_size = gesmr_dict["population"]
        self.epochs = gesmr_dict["epochs"]
        self.k_groups = gesmr_dict["k_groups"]
        self.group_size = self.population_size // self.k_groups
        self.mut_range = gesmr_dict["mut_range"]
        self.individual_ratio_breed = gesmr_dict["individual_ratio_breed"]
        self.mutation_ratio_breed = gesmr_dict["mutation_ratio_breed"]
        self.mutation_ratio_mutate = gesmr_dict["mutation_ratio_mutate"]

        self.save_logs_every_n_epochs = gesmr_dict["save_logs_every_n_epochs"]
        base_log_dir = gesmr_dict["logs_path"]

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
        self.max_threads = os.cpu_count() if gesmr_dict["max_threads"] <= 0 else gesmr_dict["max_threads"]
        # end of things taken from constants_dict

        # file handling
        self.log_directory = base_log_dir + "/" + "GESMR" + str(int(time.time())) + "/"
        os.makedirs(self.log_directory, exist_ok=True)

        #self.logger = Timestamp_Logger(file_path=self.log_directory + "log.txt", log_mode='w', log_moment='a', separator='\t')
        self.neural_network_kwargs = constants_dict["neural_network"]
        self.population = [
            GESMR_Immutable_Individual(self.neural_network_kwargs,
                       self.environment_class,
                       self.training_environments_kwargs)
            for _ in range(self.population_size)
        ]
        self.mutation_population = []
        self.best_individual = self.population[0]
        self.mutations = self._init_mutations(self.mut_range, self.k_groups)

    @staticmethod
    def _init_mutations(mut_range: tuple[float, float], size: int) -> np.ndarray:
        # change_rate = (mut_range[1] / mut_range[0]) ** (1 / (size - 1))
        # mutation_factors = np.empty(size)
        # current_value = mut_range[0]
        # for i in range(size):
        #     mutation_factors[i] = current_value
        #     current_value *= change_rate
        mutation_factors = np.linspace(mut_range[0], mut_range[1], size)
        return mutation_factors


    def run(self) -> pd.DataFrame:
        """
        Runs evolutionary algorithm
        :return: pd.DataFrame with logs
        """
        quantile = [0.25, 0.5, 0.75, 0.9, 0.99]
        quantile_labels = [f"quantile_{q}" for q in quantile]
        log_list = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(individual.get_fitness)
                for individual in self.population
            ]
            _ = [future.result() for future in futures]
        self.population = sorted(self.population, key=lambda individual: individual.get_fitness(), reverse=True)

        for generation in range(self.epochs):
            print(f"Generation {generation}")
            print(f"mutations: {', '.join([f'{mutation:.3f}' for mutation in sorted(self.mutations)])}")

            individuals_to_choose_from = self.population[:int(self.population_size * self.individual_ratio_breed)]
            mutation_tuples = [
                (np.random.choice(individuals_to_choose_from), self.mutations[i // self.group_size])
                for i in range(self.population_size)
            ]


            time_start = time.perf_counter()
            # multi-threading
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                futures = [
                    executor.submit(mutation_tuple[0].copy_mutate_and_evaluate, mutation_tuple[1])
                    for mutation_tuple in mutation_tuples
                ]
                mutated_population = [future.result() for future in futures]
            # mutated_population = [
            #     individual.copy_mutate_and_evaluate()
            #     for individual in self.population
            # ]
            # end of multi-threading
            time_end = time.perf_counter()
            print(f"Time: {time_end - time_start}, mean time: {(time_end - time_start) / self.population_size / 2}, mean time using one thread: {(time_end - time_start) / self.population_size / 2 * self.max_threads}")

            deltas = np.array([individual.get_fitness() for individual in mutated_population]) - np.array([mutation_tuple[0].get_fitness() for mutation_tuple in mutation_tuples])
            deltas = np.clip(deltas, 0, None)
            deltas_per_mutation = np.array([np.max(deltas[i * self.group_size: (i + 1) * self.group_size]) for i in range(self.k_groups)])
            sorted_mutations = np.array(self.mutations[np.argsort(deltas_per_mutation)[::-1]])
            sorted_mutations = sorted_mutations[:int(self.k_groups * self.mutation_ratio_breed)]
            self.mutations = np.random.choice(sorted_mutations, self.k_groups)
            self.mutations = self.mutations * (self.mutation_ratio_mutate ** np.random.uniform(-1,1, self.k_groups))
            self.mutations = np.clip(self.mutations, self.mut_range[0], self.mut_range[1])
            self.mutations[0] = sorted_mutations[0]

            best_individual = max(self.population, key=lambda individual: individual.get_fitness())

            if best_individual.get_fitness() > self.best_individual.get_fitness():
                self.best_individual = best_individual

            self.population = sorted(
                # [best_individual] + mutated_population,
                self.population + mutated_population,
                key=lambda individual: individual.get_fitness(),
                reverse=True
            )[:self.population_size]

            fitnesses = np.array([individual.get_fitness() for individual in self.population])
            quantile_results = np.quantile(fitnesses, quantile)
            quantile_text = ", ".join([f"{quantile}: {quantile_results[i]}" for i, quantile in enumerate(quantile)])
            print(f"Mean fitness: {fitnesses.mean()}, best fitness: {self.best_individual.get_fitness()}")
            print(f"Quantiles: {quantile_text}\n\n")

            if generation % self.save_logs_every_n_epochs == 0:
                # model = self.best_individual.neural_network
                # run_basic_environment_visualization(model)
                log_list.append(
                    {
                        "generation": generation,
                        "mean_fitness": fitnesses.mean(),
                        "best_fitness": self.best_individual.get_fitness(),
                        **{label: value for label, value in zip(quantile_labels, quantile_results)}
                    }
                )

            # print(self.mutation_controller)
        log_data_frame = pd.DataFrame(log_list)

        return log_data_frame


class GESMR_Immutable_Individual:
    def __init__(self,
                 neural_network_params: Dict[str, Any],
                 environment_class: Type[Abstract_Environment],
                 environments_list_kwargs: List[Dict[str, Any]]) -> None:
        """
        Initializes Individual
        :param neural_network_params: parameters for neural network
        :param environment_class: class of environment
        :param environments_list_kwargs: list of kwargs for environments
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
            # self.param_tree_self.params = self.neural_network.get_parameters()
            # self.param_tree_self.fitness = self.fitness
        return self.fitness

    def _copy(self) -> 'GESMR_Immutable_Individual':
        new_individual = GESMR_Immutable_Individual(self.neural_network_params,
                                    self.environment_class,
                                    self.environments_kwargs)
        new_individual.neural_network.set_parameters(self.neural_network.get_parameters())
        new_individual.fitness = self.get_fitness()
        new_individual.is_fitness_calculated = self.is_fitness_calculated
        return new_individual

    def copy_mutate_and_evaluate(self, mutation_factor: float) -> 'GESMR_Immutable_Individual':
        """
        Copies, mutates and evaluates individual
        :param mutation_factor:
        """
        new_individual = self._copy()

        new_individual_params = new_individual.neural_network.get_parameters()
        GESMR_Immutable_Individual._permute(new_individual_params, mutation_factor)
        new_individual.neural_network.set_parameters(new_individual_params)
        new_individual.is_fitness_calculated = False

        new_individual.get_fitness()

        return new_individual

    @staticmethod
    def _permute(param: dict[str, Any] | np.ndarray, mut_factor: float) -> None:
        """
        Permutes parameters dictionary inplace
        """
        if isinstance(param, np.ndarray):
            param += np.random.normal(0, mut_factor, param.shape)
        else:
            for key, value in param.items():
                GESMR_Immutable_Individual._permute(value, mut_factor)

import os
import pickle
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
from src_files.Evolutionary_Algorithms.Mutation_Controllers.mutation_controllers_functions import \
    get_mutation_controller_by_name, Abstract_Mutation_Controller
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Models.Normal.Normal_model import Normal_model


class Evolutionary_Mutate_Population:
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
        self.population_size = constants_dict["Evolutionary_Mutate_Population"]["population"]
        self.epochs = constants_dict["Evolutionary_Mutate_Population"]["epochs"]
        # self.mutation_factor = constants_dict["Evolutionary_Mutate_Population"]["mutation_factor"]
        # self.use_safe_mutation = constants_dict["Evolutionary_Mutate_Population"]["use_safe_mutation"]
        # self.L1 = constants_dict["Evolutionary_Mutate_Population"]["L1"]
        # self.L2 = constants_dict["Evolutionary_Mutate_Population"]["L2"]
        self.save_logs_every_n_epochs = constants_dict["Evolutionary_Mutate_Population"]["save_logs_every_n_epochs"]
        self.max_evaluations = constants_dict["Evolutionary_Mutate_Population"]["max_evaluations"]
        base_log_dir = constants_dict["Evolutionary_Mutate_Population"]["logs_path"]

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
        self.max_threads = os.cpu_count() if constants_dict["Evolutionary_Mutate_Population"]["max_threads"] <= 0 else constants_dict["Evolutionary_Mutate_Population"]["max_threads"]
        # end of things taken from constants_dict

        # file handling
        self.log_directory = base_log_dir + "/" + "EvMuPop" + str(int(time.time())) + "/"
        # os.makedirs(self.log_directory, exist_ok=True)

        #self.logger = Timestamp_Logger(file_path=self.log_directory + "log.txt", log_mode='w', log_moment='a', separator='\t')
        self.mutation_controller = get_mutation_controller_by_name(
            constants_dict["Evolutionary_Mutate_Population"]["mutation_controller"]["name"]
        )(**constants_dict["Evolutionary_Mutate_Population"]["mutation_controller"]["kwargs"])
        self.neural_network_kwargs = constants_dict["neural_network"]
        self.population = [
            Immutable_Individual(self.neural_network_kwargs,
                       self.environment_class,
                       self.training_environments_kwargs,
                       self.mutation_controller)
                       # self.mutation_factor,
                       # use_safe_mutation=self.use_safe_mutation,
                       # L1_factor=self.L1,
                       # L2_factor=self.L2)
            for _ in range(self.population_size)
        ]
        self.best_individual = self.population[0].copy()


    def run(self) -> pd.DataFrame:
        """
        Runs evolutionary algorithm
        :return: pd.DataFrame with logs
        """
        quantile = [0.25, 0.5, 0.75, 0.9, 0.99]
        quantile_labels = [f"quantile_{q}" for q in quantile]
        log_list = []

        # with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
        #     futures = [
        #         executor.submit(individual.get_fitness)
        #         for individual in self.population
        #     ]
        #     results = [future.result() for future in futures]

        for generation in range(self.epochs):
            print(f"Generation {generation}")

            time_start = time.perf_counter()
            # multi-threading
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                futures = [
                    executor.submit(individual.copy_mutate_and_evaluate)
                    for individual in self.population
                ]
                mutated_population = [future.result() for future in futures]
            # mutated_population = [
            #     individual.copy_mutate_and_evaluate()
            #     for individual in self.population
            # ]
            # end of multi-threading
            time_end = time.perf_counter()
            print(f"Time: {time_end - time_start}, mean time: {(time_end - time_start) / self.population_size / 2}, mean time using one thread: {(time_end - time_start) / self.population_size / 2 * self.max_threads}")


            # previous_best_fitness = self.best_individual.get_fitness()

            all_population = sorted(
                self.population + mutated_population,
                key=lambda individual: individual.get_fitness(),
                reverse=True
            )
            self.population = all_population[:self.population_size]
            # for individual in all_population[self.population_size:]:
            #     individual.parent = None
            #     for child in individual.children:
            #         child.parent = None

            if self.population[0].get_fitness() > self.best_individual.get_fitness():
                # self.population[0].FIHC(self.mutation_factor, 20, self.mutation_threshold)
                self.best_individual = self.population[0].copy()
                # print(self.population[0].get_fitness())
                # if self.population[0].get_fitness() >= 1000.0:
                #     def get_deepest_parent(individual: Immutable_Individual) -> Immutable_Individual:
                #         while individual.parent is not None:
                #             individual = individual.parent
                #         return individual
                #     def extend_lists(individuals: list[Immutable_Individual]) -> list[tuple[float, list | None]]:
                #         return [
                #             (individual.get_fitness(), extend_lists(individual.children) if len(individual.children) > 0 else None)
                #             for individual in individuals
                #         ]
                #
                #     deepest_parents = {get_deepest_parent(individual) for individual in self.population}
                #     population_tree = extend_lists(list(deepest_parents))
                #     with open(f"{self.log_directory}population_tree.pkl", "wb") as file:
                #         pickle.dump(population_tree, file)




            fitnesses = np.array([individual.get_fitness() for individual in self.population])
            self.mutation_controller.commit_iteration(fitnesses)
            quantile_results = np.quantile(fitnesses, quantile)
            quantile_text = ", ".join([f"{quantile}: {quantile_results[i]}" for i, quantile in enumerate(quantile)])
            print(f"Mean fitness: {fitnesses.mean()}, best fitness: {self.best_individual.get_fitness()}")
            print(f"Quantiles: {quantile_text}\n\n")

            evaluations = self.population_size * (2 + generation)

            if generation % self.save_logs_every_n_epochs == 0:
                # model = self.best_individual.neural_network
                # run_basic_environment_visualization(model)
                log_list.append(
                    {
                        "generation": generation,
                        "mean_fitness": fitnesses.mean(),
                        "best_fitness": self.best_individual.get_fitness(),
                        "evaluations": evaluations,
                        **{label: value for label, value in zip(quantile_labels, quantile_results)}
                    }
                )

            # print(evaluations, self.max_evaluations)
            if evaluations >= self.max_evaluations:
                break

            # print(self.mutation_controller)
        log_data_frame = pd.DataFrame(log_list)

        return log_data_frame


class Immutable_Individual:
    def __init__(self,
                 neural_network_params: Dict[str, Any],
                 environment_class: Type[Abstract_Environment],
                 environments_list_kwargs: List[Dict[str, Any]],
                 mutation_controller: Abstract_Mutation_Controller) -> None:
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
        self.mutation_controller = mutation_controller
        # self.parent = parent
        # self.children = []

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

    def copy(self) -> 'Individual':
        # self_copy = self.__copy_set_same_previous()
        # new_individual = Individual(self.neural_network_params, self.environment_class, self.environments_kwargs, self_copy)
        new_individual = Immutable_Individual(self.neural_network_params,
                                    self.environment_class,
                                    self.environments_kwargs,
                                    self.mutation_controller)
        new_individual.neural_network.set_parameters(self.neural_network.get_parameters())
        new_individual.fitness = self.get_fitness()
        new_individual.is_fitness_calculated = self.is_fitness_calculated
        return new_individual

    def copy_mutate_and_evaluate(self) -> 'Individual':
        """
        Copies, mutates and evaluates individual
        :param mutation_factor:
        :param mutation_threshold: None or float - if not None, then it uses scaled mutation
        :return:
        """
        # if self.use_safe_mutation and (self.safe_mutation_factors is None or self.safe_mutation_factors_age >= self.safe_mutation_factor_max_age):
        #     self.fitness, inputs, outputs = self.environment_iterator.get_results_with_input_output(self.neural_network, 10_000)
        #     self.is_fitness_calculated = True
        #     self.safe_mutation_factors = self.neural_network.get_safe_mutation(inputs, outputs)
        #     self.safe_mutation_factors_age = 0
        new_individual = self.copy()

        # new_individual.parent = self
        new_individual.mutation_controller.mutate(new_individual, self)
        new_individual.is_fitness_calculated = False

        new_individual.get_fitness()
        # self.children.append(new_individual)

        return new_individual

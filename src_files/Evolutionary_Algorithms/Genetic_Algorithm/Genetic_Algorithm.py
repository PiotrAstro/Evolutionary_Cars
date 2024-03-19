import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

import numpy as np

from src_files.Environments.general_functions_provider import get_environment_class
from src_files.Environments_Visualization.Basic_Environment_Visualization import run_basic_environment_visualization
from src_files.Evolutionary_Algorithms.Individual import Individual


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
        seed = int(time.time())
        np.random.seed(seed)

        # things taken from constants_dict:
        self.population_size = constants_dict["Genetic_Algorithm"]["population"]
        self.epochs = constants_dict["Genetic_Algorithm"]["epochs"]
        self.mutation_factor = constants_dict["Genetic_Algorithm"]["mutation_factor"]
        self.crosses_per_epoch = constants_dict["Genetic_Algorithm"]["crosses_per_epoch"]
        self.new_individual_every_n_epochs = constants_dict["Genetic_Algorithm"]["new_individual_every_n_epochs"]
        self.save_logs_every_n_epochs = constants_dict["Genetic_Algorithm"]["save_logs_every_n_epochs"]
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
        os.makedirs(self.log_directory, exist_ok=True)

        #self.logger = Timestamp_Logger(file_path=self.log_directory + "log.txt", log_mode='w', log_moment='a', separator='\t')

        self.neural_network_kwargs = constants_dict["neural_network"]
        self.population = [
            Individual(self.neural_network_kwargs, self.environment_class, self.training_environments_kwargs)
            for _ in range(self.population_size)
        ]
        self.best_individual = self.population[0].copy()

        self.max_threads = os.cpu_count()


    def run(self) -> None:
        """
        Runs evolutionary algorithm
        :return:
        """
        for generation in range(self.epochs):
            print(f"Generation {generation}")

            if generation % self.new_individual_every_n_epochs == 0:
                self.population[random.randint(0, self.population_size - 1)] = Individual(self.neural_network_kwargs, self.environment_class, self.training_environments_kwargs)

            indecies_randomized = np.random.permutation(self.population_size)

            time_start = time.perf_counter()
            # multi-threading
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                futures = [
                    executor.submit(self.__perform_cross, indecies_randomized[i * 2], indecies_randomized[i * 2 + 1])
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

            if generation % self.save_logs_every_n_epochs == 0:
                model = self.best_individual.neural_network
                run_basic_environment_visualization(model)


    # def __evaluate_policy_with_logs(self, policy_here: Policy, environment_here: Environment_Interface) -> float:
    #     """
    #     Evaluates policy with normal python environment (not optimized, so that it takes care of logs)
    #     :param policy: policy
    #     :param environment: environment
    #     :return: reward
    #     """
    
    #     environment_here.reset()
    #     output = policy_here.calculate_one_example(environment_here.get_whole_state())
    #     action = environment_here.get_action_production_state(output)
    #     reward, _ = environment_here.react_to_action(action)
    
    #     while environment_here.is_alive():
    #         output = policy_here.calculate_one_example(environment_here.get_optimized_rnn_state())
    #         action = environment_here.get_action_production_state(output)
    #         reward_tmp, _ = environment_here.react_to_action(action)
    #         reward += reward_tmp
    
    #     return reward

    def __perform_cross(self, parent_index_1: int, parent_index_2: int) -> None:
        """
        Performs crosses, changes population in place
        :param parent_index_1: index of first parent
        :param parent_index_2: index of second parent
        """
        new_individual_1, new_individual_2 = self.population[parent_index_1].generate_crossed_scattered_individuals(self.population[parent_index_2])
        new_individual_1.mutate(self.mutation_factor)
        new_individual_2.mutate(self.mutation_factor)
        # new_individual_1 = self.population[parent_index_1].generate_crossed_mean_individual(self.population[parent_index_2])
        # new_individual_1.mutate(self.mutation_factor)
        # new_individual_1.get_fitness()

        better_individual = new_individual_1 if new_individual_1.get_fitness() > new_individual_2.get_fitness() else new_individual_2
        worse_parent_index = parent_index_1 if self.population[parent_index_1].get_fitness() < self.population[parent_index_2].get_fitness() else parent_index_2
        self.population[worse_parent_index] = better_individual


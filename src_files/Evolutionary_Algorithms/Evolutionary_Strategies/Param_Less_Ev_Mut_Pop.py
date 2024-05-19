import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

import numpy as np
import pandas as pd

from src_files.Environments.general_functions_provider import get_environment_class
from src_files.Environments_Visualization.Basic_Environment_Visualization import run_basic_environment_visualization
from src_files.Evolutionary_Algorithms.Individual import Individual
from src_files.Evolutionary_Algorithms.Mutation_Controllers.mutation_controllers_functions import \
    get_mutation_controller_by_name, Abstract_Mutation_Controller


class Param_Les_Ev_Mut_Pop:
    """
    Param_Less Evolutionary Mutate Population - does what evolutionary mutate population does, but without pop size - has quite a few populations at once
    """
    def __init__(self, constants_dict: Dict[str, Any]) -> None:
        """
        Initializes mutation only evolutionary algorithm
        :param constants_dict:
        :return:
        """
        seed = int(time.time() * 10000) % 2**32
        np.random.seed(seed)

        # universal params
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


        # things taken from constants_dict:
        self_dict = constants_dict["Param_Les_Ev_Mut_Pop"]
        self.epochs = self_dict["epochs"]
        self.save_logs_every_n_epochs = self_dict["save_logs_every_n_epochs"]
        base_log_dir = self_dict["logs_path"]


        assert self.environment_class is not None
        self.max_threads = os.cpu_count() if self_dict["max_threads"] <= 0 else self_dict["max_threads"]
        # end of things taken from constants_dict

        # file handling
        self.log_directory = base_log_dir + "/" + "ParLessEvMutPop" + str(int(time.time())) + "/"
        os.makedirs(self.log_directory, exist_ok=True)

        #self.logger = Timestamp_Logger(file_path=self.log_directory + "log.txt", log_mode='w', log_moment='a', separator='\t')
        self.constants_dict = constants_dict
        self.self_dict = self_dict

        self.best_individual = None


    def run(self) -> pd.DataFrame:
        """
        Runs evolutionary algorithm
        :return: pd.DataFrame with logs
        """
        log_list = []
        single_populations: dict[int, Single_Population] = {}

        for generation in range(self.epochs):
            print(f"Generation {generation}")

            level = 1
            generation_tmp = generation
            while generation_tmp % 4 == 0:
                generation_tmp //= 4
                if generation_tmp == 0:
                    break
                level += 1
            real_level = level
            min_level = 1
            max_level = 1
            if single_populations:
                min_level = min(single_populations)
                max_level = max(single_populations)
                real_level += min_level - 1

            if real_level not in single_populations:
                value = 16 * 2**real_level
                single_populations[real_level] = self.create_single_population(value)
                # print("fdsa")

            print(f"Population: {real_level} pop size: {single_populations[real_level].population_size}")
            single_populations[real_level].generation()

            if not self.best_individual or single_populations[real_level].best_individual.get_fitness() > self.best_individual.get_fitness():
                self.best_individual = single_populations[real_level].best_individual.copy()

            for i, value in single_populations.items():
                print(f"{i}: {value}")
            for i in range(real_level + 1, max_level + 1):
                individuals_copy = [individual.copy() for individual in single_populations[real_level].population]
                single_populations[i].add_individuals(individuals_copy)

            # for i in range(min_level, real_level):
            #     # if single_populations[real_level - 1].quantiles[1] < single_populations[real_level].quantiles[1]:
            #     if i in single_populations and single_populations[i].best_individual.get_fitness() < single_populations[real_level].best_individual.get_fitness():
            #     # if single_populations[i].mean_fitness < single_populations[real_level].mean_fitness:
            #         single_populations.pop(i)
            #         print(f"\n\n\nRemoved level {i}\n\n\n")
            #     else:
            #         break
            i = min_level
            # if i != real_level and single_populations[i].best_individual.get_fitness() < single_populations[real_level].best_individual.get_fitness():
            if single_populations[i].mean_fitness < single_populations[real_level].mean_fitness:
            # if i != real_level and single_populations[i].mean_fitness < single_populations[real_level].mean_fitness:
                single_populations.pop(i)
                print(f"\n\n\nRemoved level {i}\n\n\n")


            if generation % self.save_logs_every_n_epochs == 0:
                model = self.best_individual.neural_network
                run_basic_environment_visualization(model)
                log_list.append(
                    {
                        "generation": generation,
                        "best_fitness": self.best_individual.get_fitness(),
                    }
                )
        log_data_frame = pd.DataFrame(log_list)

        return log_data_frame

    def create_single_population(self, size: int) -> 'Single_Population':
        mutation_controller = get_mutation_controller_by_name(
            self.self_dict["mutation_controller"]["name"]
        )(**self.self_dict["mutation_controller"]["kwargs"])
        neural_network_kwargs = self.constants_dict["neural_network"]
        population = [
            Individual(neural_network_kwargs,
                       self.environment_class,
                       self.training_environments_kwargs,
                       mutation_controller)
            for _ in range(size)
        ]
        max_threads = self.self_dict["max_threads"]
        return Single_Population(population, mutation_controller, max_threads)


class Single_Population:
    quantile_values = (0.25, 0.5, 0.75)

    def __init__(self, population: list[Individual], mutation_controller: Abstract_Mutation_Controller, threads_num: int):
        self.population = population
        self.population_size = len(population)
        self.mutation_controller = mutation_controller
        self.best_individual = population[0].copy()
        self.median_individual = self.best_individual
        self.quantiles = np.array([0.0 for _ in self.quantile_values])
        self.threads_num = threads_num
        self.mean_fitness = 0.0

    def add_individuals(self, new_individuals: list[Individual]):
        fitness_dict = {individual.get_fitness(): individual for individual in self.population}
        for new_individual in new_individuals:
            if not new_individual.get_fitness() in new_individuals:
                fitness_dict[new_individual.get_fitness()] = new_individual
        self.population = sorted(
            fitness_dict.values(),
            key=lambda individual: individual.get_fitness(),
            reverse=True
        )[:self.population_size]

        if self.population[0].get_fitness() > self.best_individual.get_fitness():
            # self.population[0].FIHC(self.mutation_factor, 20, self.mutation_threshold)
            self.best_individual = self.population[0].copy()

        fitnesses = np.array([individual.get_fitness() for individual in self.population])
        self.mean_fitness = np.mean(fitnesses)
        self.quantiles = np.quantile(fitnesses, (0.25, 0.5, 0.75))

    def generation(self):
        time_start = time.perf_counter()
        # multi-threading
        with ThreadPoolExecutor(max_workers=self.threads_num) as executor:
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
        print(
            f"Time: {time_end - time_start}, mean time: {(time_end - time_start) / self.population_size}, mean time using one thread: {(time_end - time_start) / self.population_size * self.threads_num}")

        previous_best_fitness = self.best_individual.get_fitness()
        self.mutation_controller.commit_iteration(previous_best_fitness)

        self.add_individuals(mutated_population)

        quantile_text = ", ".join([f"{quantile}: {self.quantiles[i]}" for i, quantile in enumerate(self.quantile_values)])
        print(f"Mean fitness: {self.mean_fitness}, best fitness: {self.best_individual.get_fitness()}")
        print(f"Quantiles: {quantile_text}\n\n")

    def __str__(self) -> str:
        return str(self.population_size)

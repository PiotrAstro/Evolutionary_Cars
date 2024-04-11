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
    get_mutation_controller_by_name


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
        seed = int(time.time())
        np.random.seed(seed)

        # things taken from constants_dict:
        self.population_size = constants_dict["Evolutionary_Mutate_Population"]["population"]
        self.epochs = constants_dict["Evolutionary_Mutate_Population"]["epochs"]
        # self.mutation_factor = constants_dict["Evolutionary_Mutate_Population"]["mutation_factor"]
        # self.use_safe_mutation = constants_dict["Evolutionary_Mutate_Population"]["use_safe_mutation"]
        # self.L1 = constants_dict["Evolutionary_Mutate_Population"]["L1"]
        # self.L2 = constants_dict["Evolutionary_Mutate_Population"]["L2"]
        self.save_logs_every_n_epochs = constants_dict["Evolutionary_Mutate_Population"]["save_logs_every_n_epochs"]
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
        os.makedirs(self.log_directory, exist_ok=True)

        #self.logger = Timestamp_Logger(file_path=self.log_directory + "log.txt", log_mode='w', log_moment='a', separator='\t')
        self.mutation_controller = get_mutation_controller_by_name(
            constants_dict["Evolutionary_Mutate_Population"]["mutation_controller"]["name"]
        )(**constants_dict["Evolutionary_Mutate_Population"]["mutation_controller"]["kwargs"])
        self.neural_network_kwargs = constants_dict["neural_network"]
        self.population = [
            Individual(self.neural_network_kwargs,
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

            self.mutation_controller.commit_iteration()

            self.population = sorted(
                self.population + mutated_population,
                key=lambda individual: individual.get_fitness(),
                reverse=True
            )[:self.population_size]

            if self.population[0].get_fitness() > self.best_individual.get_fitness():
                # self.population[0].FIHC(self.mutation_factor, 20, self.mutation_threshold)
                self.best_individual = self.population[0].copy()
                # if self.population[0].get_fitness() > 10000:
                #     deepest_parent = self.population[0].param_tree_self
                #     while deepest_parent.parent is not None:
                #         deepest_parent = deepest_parent.parent
                #     with open(self.log_directory + "best_individual_tree.pkl", "wb") as f:
                #         pickle.dump(deepest_parent, f)


            fitnesses = np.array([individual.get_fitness() for individual in self.population])
            quantile_results = np.quantile(fitnesses, quantile)
            quantile_text = ", ".join([f"{quantile}: {quantile_results[i]}" for i, quantile in enumerate(quantile)])
            print(f"Mean fitness: {fitnesses.mean()}, best fitness: {self.best_individual.get_fitness()}")
            print(f"Quantiles: {quantile_text}\n\n")

            if generation % self.save_logs_every_n_epochs == 0:
                # params = self.best_individual.neural_network.get_parameters()
                # print(f"Best individual parameters: {params}")
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

            print(self.mutation_controller)
        log_data_frame = pd.DataFrame(log_list)

        return log_data_frame

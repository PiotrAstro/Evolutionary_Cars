import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

import numpy as np

from src_files.Environments.general_functions_provider import get_environment_class
from src_files.Environments_Visualization.Basic_Environment_Visualization import run_basic_environment_visualization
from src_files.Evolutionary_Algorithms.Individual import Individual


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
        seed = int(time.time())
        np.random.seed(seed)

        # things taken from constants_dict:
        self.population_size = constants_dict["Differential_Evolution"]["population"]
        self.epochs = constants_dict["Differential_Evolution"]["epochs"]
        self.cross_prob = constants_dict["Differential_Evolution"]["cross_prob"]
        self.diff_weight = constants_dict["Differential_Evolution"]["diff_weight"]
        self.save_logs_every_n_epochs = constants_dict["Differential_Evolution"]["save_logs_every_n_epochs"]
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

            time_start = time.perf_counter()
            # multi-threading
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                futures = []
                for i in range(self.population_size):
                    base_id, id1, id2 = np.random.choice(self.population_size, 3, replace=False)
                    base_params = self.population[base_id].neural_network.get_parameters()
                    id1_params = self.population[id1].neural_network.get_parameters()
                    id2_params = self.population[id2].neural_network.get_parameters()
                    futures.append(executor.submit(self.population[i].differential_evolution_one_epoch, base_params, id1_params, id2_params, self.cross_prob, self.diff_weight))

                results = [future.result() for future in futures]
            # for i in range(self.crosses_per_epoch):
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

            if generation % self.save_logs_every_n_epochs == 0:
                model = self.best_individual.neural_network
                run_basic_environment_visualization(model)

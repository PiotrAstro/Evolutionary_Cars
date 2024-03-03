import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Tuple, List, Mapping

import numpy as np

from src_files.Environments.general_functions_provider import get_environment_class
from src_files.Environments_Visualization.Basic_Environment_Visualization import run_basic_environment_visualization
from src_files.Evolutionary_Algorithms.Individual import Individual


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
        os.makedirs(self.log_directory, exist_ok=True)

        # self.logger = Timestamp_Logger(file_path=self.log_directory + "log.txt", log_mode='w', log_moment='a', separator='\t')

        self.neural_network_kwargs = constants_dict["neural_network"]
        self.individual = Individual(self.neural_network_kwargs, self.environment_class, self.training_environments_kwargs)

        self.max_threads = os.cpu_count()


    def run(self) -> None:
        """
        Runs evolutionary algorithm
        :return:
        """
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

            if generation % self.stats_every_n_epochs == 0:
                model = self.individual.neural_network
                run_basic_environment_visualization(model)

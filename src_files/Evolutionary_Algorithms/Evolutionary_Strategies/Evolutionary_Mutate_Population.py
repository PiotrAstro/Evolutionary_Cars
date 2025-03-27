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
from src_files.Neural_Network.Pytorch.Normal_Model import Normal_Model_Pytorch
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
        self.log_directory = base_log_dir + "/" + "EvMuPop" + str(int(time.time() * 1000)) + "/"
        # os.makedirs(self.log_directory, exist_ok=True)

        #self.logger = Timestamp_Logger(file_path=self.log_directory + "log.txt", log_mode='w', log_moment='a', separator='\t')
        self.mutation_controller = get_mutation_controller_by_name(
            constants_dict["Evolutionary_Mutate_Population"]["mutation_controller"]["name"]
        )(**constants_dict["Evolutionary_Mutate_Population"]["mutation_controller"]["kwargs"])
        self.constants_dict = constants_dict
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

    def get_2_individuals_tournament(self, prob_choose_random: float = 0.1, choose_from: int = 10) -> tuple['Immutable_Individual', 'Immutable_Individual']:
        """
        Returns 2 individuals from population using tournament selection
        :param prob_choose_random: float - probability of choosing random individual
        :return: tuple[Immutable_Individual, Immutable_Individual]
        """
        choices = np.random.choice(self.population, choose_from * 10, replace=False)
        choices1 = choices[:choose_from]
        choices2 = choices[choose_from:]

        if np.random.rand() < prob_choose_random:
            individual_1 = random.choice(choices1)
        else:
            individual_1 = max(choices1, key=lambda individual: individual.get_fitness())

        if np.random.rand() < prob_choose_random:
            individual_2 = random.choice(choices2)
        else:
            individual_2 = max(choices2, key=lambda individual: individual.get_fitness())

        return individual_1, individual_2


    def run(self) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        """
        Runs evolutionary algorithm
        :return: pd.DataFrame with logs
        """
        quantile = [0.25, 0.5, 0.75, 0.9, 0.99]
        quantile_labels = [f"quantile_{q}" for q in quantile]
        log_list = []
        final_model = None

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

            # self.population.append(
            #     Immutable_Individual.crossover(*self.get_2_individuals_tournament())
            # )

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
                self.best_individual.local_search()
                # save_file_path = self.log_directory + f"best_individual_gen{generation}_{int(time.time())}_f{self.best_individual.get_fitness():.1f}.pkl"
                # final_model = {
                #     "universal_kwargs": self.constants_dict["environment"]["universal_kwargs"],
                #     "neural_network_params": self.best_individual.neural_network.get_parameters()
                # }
                # with open(save_file_path, "wb") as file:
                #     pickle.dump(final_model, file)
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
                model = self.best_individual.neural_network
                run_basic_environment_visualization(model)
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

        return log_data_frame, final_model


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
            # self.fitness = sum([environment.p_get_trajectory_results(self.neural_network) for environment in self.environments])
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



    def local_search(self) -> 'Immutable_Individual':
        """
        Local search
        """

        states, actions = [], []
        ready_fitness = 0.0
        for environment in self.environments:
            # tmp_fitness, tmp_states, tmp_actions = self._ls_optimize_trajectory(environment)
            environment.p_reset()
            tmp_fitness, tmp_states, tmp_actions = environment.p_get_trajectory_logs(self.neural_network)

            ready_fitness += tmp_fitness
            states.append(tmp_states)
            actions.append(tmp_actions)
        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)

        old_fitness = self.get_fitness()
        previous_nn = self.neural_network
        new_nn = Normal_model(**self.neural_network_params)
        self.neural_network = new_nn

        self._learn(actions, states)

        self.is_fitness_calculated = False
        new_fitness = self.get_fitness()

        if old_fitness > new_fitness:
            self.neural_network = previous_nn
            self.fitness = old_fitness
        print(f"Old fitness: {old_fitness:.2f}, optimized_fitness: {ready_fitness:.2f}, new fitness: {new_fitness:.2f}")


    def _ls_optimize_trajectory(self, environment: Abstract_Environment) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Optimizes trajectory via local search
        :param environment:
        :return:
        """
        TRIES = 3

        ready_states = []
        ready_actions = []
        ready_fitness = 0.0
        original_fitness, original_states, original_actions = environment.p_get_trajectory_logs(self.neural_network)
        environment.p_reset()
        environment_safe = environment.p_get_safe_data()
        while len(original_states) > 0:
            for i in range(TRIES):
                environment.p_load_safe_data(environment_safe)

                new_action = np.zeros_like(original_actions[0], dtype=np.float32)# np.array(original_actions[0], dtype=np.float32, copy=True)
                new_action[np.random.randint(3)] = 1.0
                new_action[3 + np.random.randint(3)] = 1.0
                # new_action[-1] = np.random.uniform(-1.0, 1.0)
                # rand_tmp = np.random.uniform(-1.0, 1.0, 3)
                # rand_tmp = rand_tmp - rand_tmp.min()
                # rand_tmp = rand_tmp / rand_tmp.sum()
                # new_action[:3] += rand_tmp
                # new_action[:3] /= 2
                # new_action[-1] = np.clip(new_action[-1] + np.random.uniform(-0.5, 0.5), -1.0, 1.0)


                one_state = environment.p_get_state()
                one_fitness = environment.p_react(new_action)
                new_fitness, new_states, new_actions = environment.p_get_trajectory_logs(self.neural_network)
                new_fitness += one_fitness + ready_fitness
                if new_fitness > original_fitness:
                    original_fitness = new_fitness
                    original_states = new_states
                    original_actions = new_actions
                    ready_states.append(one_state)
                    ready_actions.append(new_action)
                    break
                elif i == TRIES - 1:
                    ready_states.append(original_states[0])
                    ready_actions.append(original_actions[0])
                    original_states = original_states[1:]
                    original_actions = original_actions[1:]

            environment.p_load_safe_data(environment_safe)
            one_final_fitness = environment.p_react(ready_actions[-1])
            ready_fitness += one_final_fitness
            environment_safe = environment.p_get_safe_data()

        return ready_fitness, np.array(ready_states), np.array(ready_actions)

    def _learn(self, actions: np.ndarray, states: np.ndarray) -> None:
        """
        Modify neural network
        :param self:
        :param actions:
        :param strates:
        :param neural_network:
        :return:
        """
        BATCH_SIZE = 64
        LOSS = [("KL_Divergence", 3), ("KL_Divergence", 3)]
        LR = 0.001
        GENERATIONS = 50

        permutation = np.random.permutation(len(states))
        actions = actions[permutation]
        states = states[permutation]

        actions_argmax_steering = np.argmax(actions[:, 0:3], axis=1)
        actions_argmax_acceleration = np.argmax(actions[:, 3:6], axis=1)
        actions = np.zeros_like(actions, dtype=np.float32)
        actions[np.arange(len(actions_argmax_steering)), actions_argmax_steering] = 1.0
        actions[np.arange(len(actions_argmax_acceleration)), 3 + actions_argmax_acceleration] = 1.0

        section_number = len(states) // BATCH_SIZE
        if section_number == 0:
            section_number = 1

        batched_states = np.array_split(states, section_number)
        batched_actions = np.array_split(actions, section_number)

        for _ in range(GENERATIONS):
            for batch_states, batch_actions in zip(batched_states, batched_actions):
                self.neural_network.backward_SGD(batch_states, batch_actions, LR, LOSS)

        results = self.neural_network.p_forward_pass(states)
        actions_argmax_steering_tmp = np.argmax(results[:, 0:3], axis=1)
        actions_argmax_acceleration_tmp = np.argmax(results[:, 3:6], axis=1)
        accuracy_steering = (actions_argmax_steering_tmp == actions_argmax_steering).mean()
        accuracy_acceleration = (actions_argmax_acceleration_tmp == actions_argmax_acceleration).mean()
        print(f"Accuracy steering: {accuracy_steering:.2f}, acceleration: {accuracy_acceleration:.2f}")

    def _collect_reset_trajectories(self) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """
        Collects trajectories from environments
        :return: tuple[float, np.ndarray, np.ndarray]
        """
        trajectories = []
        for environment in self.environments:
            environment.p_reset()
            trajectories.append(environment.p_get_trajectory_logs(self.neural_network))
            environment.p_reset()
        return trajectories

    @staticmethod
    def crossover(parent1: 'Immutable_Individual', parent2: 'Immutable_Individual') -> 'Immutable_Individual':
        """
        Crossover
        :param parent1:
        :param parent2:
        :return:
        """
        trajectories1 = parent1._collect_reset_trajectories()
        trajectories2 = parent2._collect_reset_trajectories()

        new_individual = parent1.copy()
        new_individual.neural_network = Normal_model(**parent1.neural_network_params)

        optimal_trajectories = [
            trajectory_1 if trajectory_1[0] > trajectory_2[0] else trajectory_2
            for trajectory_1, trajectory_2 in zip(trajectories1, trajectories2)
        ]

        states = np.concatenate([trajectory[1] for trajectory in optimal_trajectories], axis=0)
        actions = np.concatenate([trajectory[2] for trajectory in optimal_trajectories], axis=0)

        new_individual._learn(actions, states)
        new_individual.is_fitness_calculated = False
        print(f"Parent1 fitness: {parent1.get_fitness()}, parent2 fitness: {parent2.get_fitness()}, new fitness: {new_individual.get_fitness()}")

        return new_individual


from concurrent.futures import ThreadPoolExecutor
from typing import Mapping, Any, List, Dict, Type, Tuple, Optional

import numpy as np

from src_files.MyMath import MyMath
from src_files.Environments.Abstract_Environment.Abstract_Environment import Abstract_Environment
from src_files.Environments.Abstract_Environment.Abstract_Environment_Iterator import Abstract_Environment_Iterator
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Models.Normal.Normal_model import Normal_model


class Individual:
    def __init__(self, neural_network_params: Dict[str, Any], environment_class: Type[Abstract_Environment], environments_list_kwargs: List[Dict[str, Any]]) -> None:
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
        return self.fitness

    def generate_crossed_scattered_individuals(self, other: 'Individual') -> Tuple['Individual', 'Individual']:
        """
        Generates two new individuals, which are scattered cross of two individuals
        :param other: other individual
        :return: two new individuals
        """
        self_policy_params = self.neural_network.get_parameters()
        other_policy_params = other.neural_network.get_parameters()
        self.__cross_scattered_policy(self_policy_params, other_policy_params)
        new_individual1 = Individual(self.neural_network_params, self.environment_class, self.environments_kwargs)
        new_individual1.neural_network.set_parameters(self_policy_params)
        new_individual2 = Individual(self.neural_network_params, self.environment_class, self.environments_kwargs)
        new_individual2.neural_network.set_parameters(other_policy_params)
        return new_individual1, new_individual2

    def generate_crossed_mean_individual(self, other: 'Individual') -> 'Individual':
        """
        Generates new individual, which is mean of two individuals
        :param other: other individual
        :return: new individual
        """
        new_policy_params = self.neural_network.get_parameters()
        other_policy_params = other.neural_network.get_parameters()
        self.__cross_mean_policy(new_policy_params, other_policy_params)
        new_individual = Individual(self.neural_network_params, self.environment_class, self.environments_kwargs)
        new_individual.neural_network.set_parameters(new_policy_params)
        return new_individual

    def mutate(self, mutation_factor: float, mutation_threshold: Optional[float] = None) -> None:
        """
        Mutates individual inplace, if mutation_threshold is not None, then it uses scaled mutation
        :param mutation_factor:
        :param mutation_threshold: None or float
        :return:
        """
        params = self.neural_network.get_parameters()
        self.__permute_params_dict(params, mutation_factor, mutation_threshold=mutation_threshold)
        self.neural_network.set_parameters(params)
        self.is_fitness_calculated = False

    def copy_mutate_and_evaluate(self, mutation_factor: float, mutation_threshold: Optional[float] = None) -> 'Individual':
        """
        Copies, mutates and evaluates individual
        :param mutation_factor:
        :param mutation_threshold: None or float - if not None, then it uses scaled mutation
        :return:
        """
        new_individual = self.copy()
        new_individual.mutate(mutation_factor, mutation_threshold=mutation_threshold)
        new_individual.get_fitness()
        return new_individual

    def copy_mutate_and_evaluate_other_self(self, mutation_factor: float, mutation_threshold: Optional[float] = None) -> 'Individual':
        """
        Copies, mutates and evaluates individual, firstly self is evaluated, then mutated
        :param mutation_factor:
        :param mutation_threshold: None or float - if not None, then it uses scaled mutation
        :return:
        """
        self.get_fitness()
        return self.copy_mutate_and_evaluate(mutation_factor, mutation_threshold=mutation_threshold)

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

        fitnesses = np.array([individual.get_fitness() for individual in individuals_mutated])
        fitnesses_normalised = (fitnesses - np.mean(fitnesses)) / np.std(fitnesses)
        multiply_factor = alpha_learning_rate / (number_of_individuals * sigma_change)
        change_amount = fitnesses_normalised * multiply_factor
        other_policies_params = [individual.neural_network.get_parameters() for individual in individuals_mutated]
        self_policy_params = self.neural_network.get_parameters()
        self.__actualise_policy_dict_params(self_policy_params, other_policies_params, change_amount, [])
        self.neural_network.set_parameters(self_policy_params)

        self.is_fitness_calculated = False

        return fitnesses

    def differential_evolution_one_epoch(self,
                                         base_params: Mapping[str, Any],
                                         individual_1_params: Mapping[str, Any],
                                         individual_2_params: Mapping[str, Any],
                                         cross_prob: float,
                                         diff_weight: float) -> bool:
        """
        Performs one step of differential evolution, modifies self inplace, if modification is better than base individual, returns True, otherwise False
        :param base_params:
        :param individual_1_params:
        :param individual_2_params:
        :param cross_prob:
        :param diff_weight:
        :return: True if change accepted, False otherwise
        """
        original_params = self.neural_network.get_parameters()
        original_fitness = self.get_fitness()

        new_params = self.neural_network.get_parameters()
        self.__diff_evolution_iteration(new_params, base_params, individual_1_params, individual_2_params, cross_prob, diff_weight)

        self.neural_network.set_parameters(new_params)
        self.is_fitness_calculated = False

        if self.get_fitness() > original_fitness:
            return True
        else:
            self.neural_network.set_parameters(original_params)
            self.is_fitness_calculated = True
            self.fitness = original_fitness
            return False

    def copy(self) -> 'Individual':
        new_individual = Individual(self.neural_network_params, self.environment_class, self.environments_kwargs)
        params = self.neural_network.get_parameters()
        new_individual.neural_network.set_parameters(params)
        new_individual.fitness = self.fitness
        new_individual.is_fitness_calculated = self.is_fitness_calculated
        return new_individual

    def __diff_evolution_iteration(self,
                                   self_params: Mapping[str, Any],
                                    base_params: Mapping[str, Any],
                                    individual_1_params: Mapping[str, Any],
                                    individual_2_params: Mapping[str, Any],
                                    cross_prob: float,
                                    diff_weight: float) -> None:
        """
        Performs Differential Evolution on np.array, iterates fthrwo params
        :param self_params:
        :param base_params:
        :param individual_1_params:
        :param individual_2_params:
        :param cross_prob:
        :param diff_weight:
        :return:
        """
        for key, value in self_params.items():
            if isinstance(value, dict):
                self.__diff_evolution_iteration(value, base_params[key], individual_1_params[key], individual_2_params[key], cross_prob, diff_weight)
            elif isinstance(value, np.ndarray):
                cross_mask = np.random.rand(*value.shape) < cross_prob
                self_params[key][cross_mask] = base_params[key][cross_mask] + diff_weight * (individual_1_params[key][cross_mask] - individual_2_params[key][cross_mask])


    def __cross_mean_policy(self, params_to_change: Mapping[str, Any], other_params: Mapping[str, Any]) -> None:
        """
        Changes mean policy dictionary parameters, params_to_change = (params_to_change + other_params) / 2
        :param params_to_change: current main policy part of params
        :param other_params: params from other individual
        :param key_list: list of keys to get to the parameter
        """
        for key, value in params_to_change.items():
            if isinstance(value, dict):
                self.__cross_mean_policy(value, other_params[key])
            elif isinstance(value, np.ndarray):
                params_to_change[key] = (value + other_params[key]) / 2



    def __actualise_policy_dict_params(self, main_policy: Mapping[str, Any], other_policies: List[Mapping[str, Any]], change_amount: np.ndarray, key_list: List[str]) -> None:
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
                self.__actualise_policy_dict_params(value, other_policies, change_amount, key_list)
            elif isinstance(value, np.ndarray):
                param_array_double = value.astype(np.float64)
                for i in range(len(change_amount)):
                    param_array_double += change_amount[i] * self.__get_element_from_dict(other_policies[i], key_list)
                main_policy[key] = param_array_double.astype(np.float32)
            key_list.pop()

    def __get_element_from_dict(self, dictionary: Mapping[str, Any], key_list: List[str]) -> Any:
        """
        Gets element from dictionary by key list
        :param dictionary:
        :param key_list:
        :return:
        """
        for key in key_list:
            dictionary = dictionary[key]
        return dictionary

    def __cross_scattered_policy(self, params1: Mapping[str, Any], params2: Mapping[str, Any]) -> None:
        """
        changes params_to_change so that on random places it swapes params1 and params2
        :param params1: first policy part of params
        :param params2: second policy part of params
        """
        for key, value in params1.items():
            if isinstance(value, dict):
                self.__cross_scattered_policy(value, params2[key])
            elif isinstance(value, np.ndarray):
                mask = np.random.randint(0, 2, value.shape, dtype=np.bool_)
                params_2_mask_copy = np.array(params2[key][mask])
                params2[key][mask] = value[mask]
                params1[key][mask] = params_2_mask_copy

    def __permute_params_dict(self, param_dict: Mapping[str, Any], sigma: float, mutation_threshold: Optional[float] = None) -> None:
        """
        Permutes parameters dictionary, used with __permute_policy
        :param param_dict:
        :param sigma: standard deviation of normal distribution
        :param mutation_threshold: None or float - if not None, then it uses scaled mutation
        :return:
        """
        for key, value in param_dict.items():
            if isinstance(value, dict):
                self.__permute_params_dict(value, sigma, mutation_threshold=mutation_threshold)
            elif isinstance(value, np.ndarray):
                if mutation_threshold is None:
                    value += np.random.normal(0, sigma, value.shape)
                else:
                    MyMath.mutate_array_scaled(value, sigma, mutation_threshold)

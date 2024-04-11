from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Dict, Type, Tuple, Optional

import numpy as np

from src_files.Evolutionary_Algorithms.Mutation_Controllers.mutation_controllers_functions import \
    Abstract_Mutation_Controller
from src_files.MyMath import MyMath
from src_files.Environments.Abstract_Environment.Abstract_Environment import Abstract_Environment
from src_files.Environments.Abstract_Environment.Abstract_Environment_Iterator import Abstract_Environment_Iterator
from src_files.MyMath.MyMath import safe_mutate_inplace
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Models.Normal.Normal_model import Normal_model


class Individual_Params_Tree:
    def __init__(self, params: Dict[str, Any], parent: Optional['Individual_Params_Tree']) -> None:
        self.params = params
        self.parent = parent
        self.fitness = 0.0
        self.children: List['Individual_Params_Tree'] = []

    def add_child(self, new_child: 'Individual_Params_Tree') -> None:
        self.children.append(new_child)





# Because of mutation_factor change, currently only works normal mutation!
# L1 and L2 factors are implemented only for mutation_threshold = None
class Individual:
    def __init__(self,
                 neural_network_params: Dict[str, Any],
                 environment_class: Type[Abstract_Environment],
                 environments_list_kwargs: List[Dict[str, Any]],
                 mutation_controller: Abstract_Mutation_Controller,
                 parent: Optional[Individual_Params_Tree] = None) -> None:
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
        self.param_tree_self = None
        # self.param_tree_self = Individual_Params_Tree(self.neural_network.get_parameters(), parent)
        # if parent is not None:
        #     parent.add_child(self.param_tree_self)

        # self.params_to_look_at = self.neural_network.get_parameters()

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
        new_individual = Individual(self.neural_network_params,
                                    self.environment_class,
                                    self.environments_kwargs,
                                    self.mutation_controller,)
        new_individual.neural_network.set_parameters(self.neural_network.get_parameters())
        new_individual.fitness = self.fitness
        new_individual.is_fitness_calculated = self.is_fitness_calculated
        return new_individual

    def mutate(self) -> int:
        """
        Mutates individual inplace, returns id in mutation_controller
        :param mutation_factor:
        :param mutation_threshold: None or float
        :return:
        """
        params = self.neural_network.get_parameters()
        id_mut_contr = self.mutation_controller.mutate(params)
        self.neural_network.set_parameters(params)
        self.is_fitness_calculated = False
        return id_mut_contr

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

        # mutation factor change
        # tmp_change_rate = 0.2
        # mutation_factor_change = np.random.normal(0, tmp_change_rate)
        # if mutation_factor_change > 0:
        #     mutation_factor_change /= 1 - tmp_change_rate
        # mutation_factor_change += 1
        # new_individual.mutation_factor = self.mutation_factor * mutation_factor_change

        id_in_controller = new_individual.mutate()
        new_fitness = new_individual.get_fitness()

        if new_fitness > self.get_fitness():
            self.mutation_controller.mutation_better_than_parent(id_in_controller, self.fitness, new_fitness)

        return new_individual


########################################################################################################################
###################################    OLD FUNCTIONS   ################################################################

    # def _permute_params_dict(self, param_dict: Dict[str, Any], safe_mutation_factors: Dict[str, Any] | None = None) -> None:
    #     """
    #     Permutes parameters dictionary, used with __permute_policy
    #     :param param_dict:
    #     :param sigma: standard deviation of normal distribution
    #     :param mutation_threshold: None or float - if not None, then it uses scaled mutation
    #     :return:
    #     """
    #     for key, value in param_dict.items():
    #         if isinstance(value, dict):
    #             self._permute_params_dict(value, safe_mutation_factors[key] if safe_mutation_factors is not None else None)
    #         elif isinstance(value, np.ndarray):
    #             value += np.random.normal(0, 0.1, value.shape)
    #             # if safe_mutation_factors is None or self.use_safe_mutation is False:
    #             #     if len(value.shape) == 1:
    #             #         L1_here = 0
    #             #         L2_here = 0
    #             #     else:
    #             #         L1_here = self.L1_factor
    #             #         L2_here = self.L2_factor
    #             #     value += np.random.normal(0, self.mutation_factor, value.shape) - L1_here * np.sign(value) - L2_here * value
    #             # else:
    #             #     # sigmas = self.mutation_factor * np.clip(safe_mutation_factors[key], 0.01, 10.0)
    #             #     min_value = np.min(safe_mutation_factors[key])
    #             #     max_value = np.max(safe_mutation_factors[key])
    #             #     normalized = (safe_mutation_factors[key] - min_value) / (max_value - min_value)
    #             #     sigmas = self.mutation_factor / (0.2 + normalized)
    #             #     value += np.random.normal(0, sigmas, value.shape) - self.L1_factor * np.sign(value) - self.L2_factor * value
    #             #     # safe_mutate_inplace(value, self.mutation_factor, safe_mutation_factors[key], 0.0001, 1000.0, self.L1_factor, self.L2_factor)


    # def generate_crossed_scattered_individuals(self, other: 'Individual') -> Tuple['Individual', 'Individual']:
    #     """
    #     Generates two new individuals, which are scattered cross of two individuals
    #     :param other: other individual
    #     :return: two new individuals
    #     """
    #     self_policy_params = self.neural_network.get_parameters()
    #     other_policy_params = other.neural_network.get_parameters()
    #     self._cross_scattered_policy(self_policy_params, other_policy_params)
    #     new_individual1 = Individual(self.neural_network_params, self.environment_class, self.environments_kwargs)
    #     new_individual1.neural_network.set_parameters(self_policy_params)
    #     new_individual2 = Individual(self.neural_network_params, self.environment_class, self.environments_kwargs)
    #     new_individual2.neural_network.set_parameters(other_policy_params)
    #     return new_individual1, new_individual2

    # def generate_crossed_mean_individual(self, other: 'Individual') -> 'Individual':
    #     """
    #     Generates new individual, which is mean of two individuals
    #     :param other: other individual
    #     :return: new individual
    #     """
    #     new_policy_params = self.neural_network.get_parameters()
    #     other_policy_params = other.neural_network.get_parameters()
    #     self.__cross_mean_policy(new_policy_params, other_policy_params)
    #     new_individual = Individual(self.neural_network_params, self.environment_class, self.environments_kwargs)
    #     new_individual.neural_network.set_parameters(new_policy_params)
    #     return new_individual

    # def evolutionary_strategy_one_epoch(self, number_of_individuals: int, sigma_change: float, alpha_learning_rate: float, num_of_processes: int) -> np.ndarray:
    #     """
    #     It performs one step of evolutionary strategy, modifies self inplace
    #     :param number_of_individuals:
    #     :param sigma_change:
    #     :param alpha_learning_rate:
    #     :param num_of_processes:
    #     :return: fitnesses of all mutated individuals
    #     """
    #
    #     # multi-threading
    #     with ThreadPoolExecutor(max_workers=num_of_processes) as executor:
    #         futures = [
    #             executor.submit(self.copy_mutate_and_evaluate)
    #             for _ in range(number_of_individuals)
    #         ]
    #         individuals_mutated = [future.result() for future in futures]
    #     # end of multi-threading
    #
    #     fitnesses = np.array([individual.get_fitness() for individual in individuals_mutated], dtype=float)
    #     fitnesses_normalised = (fitnesses - np.mean(fitnesses)) / np.std(fitnesses)
    #     multiply_factor = alpha_learning_rate / (number_of_individuals * sigma_change)
    #     change_amount = fitnesses_normalised * multiply_factor
    #     other_policies_params = [individual.neural_network.get_parameters() for individual in individuals_mutated]
    #     self_policy_params = self.neural_network.get_parameters()
    #     self._actualise_policy_dict_params(self_policy_params, other_policies_params, change_amount, [])
    #     self.neural_network.set_parameters(self_policy_params)
    #
    #     self.is_fitness_calculated = False
    #
    #     return fitnesses

    # def differential_evolution_one_epoch(self,
    #                                      base_params: Dict[str, Any],
    #                                      individual_1_params: Dict[str, Any],
    #                                      individual_2_params: Dict[str, Any],
    #                                      cross_prob: float,
    #                                      diff_weight: float) -> bool:
    #     """
    #     Performs one step of differential evolution, modifies self inplace, if modification is better than base individual, returns True, otherwise False
    #     :param base_params:
    #     :param individual_1_params:
    #     :param individual_2_params:
    #     :param cross_prob:
    #     :param diff_weight:
    #     :return: True if change accepted, False otherwise
    #     """
    #     original_params = self.neural_network.get_parameters()
    #     original_fitness = self.get_fitness()
    #
    #     new_params = self.neural_network.get_parameters()
    #     self.__diff_evolution_iteration(new_params, base_params, individual_1_params, individual_2_params, cross_prob, diff_weight)
    #
    #     self.neural_network.set_parameters(new_params)
    #     self.is_fitness_calculated = False
    #
    #     if self.get_fitness() > original_fitness:
    #         return True
    #     else:
    #         self.neural_network.set_parameters(original_params)
    #         self.is_fitness_calculated = True
    #         self.fitness = original_fitness
    #         return False

    # def FIHC(self, mutation_factor: float, max_checks: int, mutation_threshold: Optional[float] = None) -> None:
    #     """
    #     Performs First Improvement Hill Climbing, modifies self inplace, takes into consideration columns of parameters (so changes each neuron)
    #     :param mutation_factor:
    #     :param max_checks: number of checks of different mutation combinations
    #     :param mutation_threshold: None or float - if not None, then it uses scaled mutation
    #     :return:
    #     """
    #     def get_list_of_things_to_consider(params: Dict[str, Any], so_far_list: List[np.ndarray]) -> List[np.ndarray]:
    #         """
    #         Gets list of things to consider, where each element is column of parameters from params
    #         :param params:
    #         :return:
    #         """
    #         for key, value in params.items():
    #             if isinstance(value, dict):
    #                 get_list_of_things_to_consider(value, so_far_list)
    #             elif isinstance(value, np.ndarray):
    #                 so_far_list.append(value)
    #         return so_far_list
    #     current_params = self.neural_network.get_parameters()
    #
    #     things_to_change = get_list_of_things_to_consider(current_params, [])
    #
    #     for param_column in things_to_change:
    #         previous_column = param_column.copy()
    #         original_fitness = self.get_fitness()
    #
    #         for _ in range(max_checks):
    #             if mutation_threshold is None:
    #                 param_column += np.random.normal(0, mutation_factor, param_column.shape)
    #             else:
    #                 MyMath.mutate_array_scaled(param_column, mutation_factor, mutation_threshold)
    #             self.neural_network.set_parameters(current_params)
    #             self.is_fitness_calculated = False
    #             new_fitness = self.get_fitness()
    #             if new_fitness > original_fitness:
    #                 self.fitness = new_fitness
    #                 break
    #             else:
    #                 self.fitness = original_fitness
    #                 param_column[:] = previous_column
    #
    #     # def get_list_of_things_to_consider(params: Dict[str, Any], so_far_list: List[np.ndarray]) -> List[np.ndarray]:
    #     #     """
    #     #     Gets list of things to consider, where each element is column of parameters from params
    #     #     :param params:
    #     #     :return:
    #     #     """
    #     #     for key, value in params.items():
    #     #         if isinstance(value, dict):
    #     #             get_list_of_things_to_consider(value, so_far_list)
    #     #         elif isinstance(value, np.ndarray):
    #     #             if len(value.shape) == 1:
    #     #                 so_far_list.append(value[:, None])
    #     #             else:
    #     #                 for i in range(value.shape[1]):
    #     #                     so_far_list.append(value[:, i, None])
    #     #     return so_far_list
    #     # current_params = self.neural_network.get_parameters()
    #     #
    #     # things_to_change = get_list_of_things_to_consider(current_params, [])
    #     #
    #     # for param_column in things_to_change:
    #     #     previous_column = param_column.copy()
    #     #     original_fitness = self.get_fitness()
    #     #
    #     #     for _ in range(max_checks):
    #     #         if mutation_threshold is None:
    #     #             param_column += np.random.normal(0, mutation_factor, param_column.shape)
    #     #         else:
    #     #             MyMath.mutate_array_scaled(param_column, mutation_factor, mutation_threshold)
    #     #         self.neural_network.set_parameters(current_params)
    #     #         self.is_fitness_calculated = False
    #     #         new_fitness = self.get_fitness()
    #     #         if new_fitness > original_fitness:
    #     #             self.fitness = new_fitness
    #     #             break
    #     #         else:
    #     #             self.fitness = original_fitness
    #     #             param_column[:] = previous_column



    # def __copy_set_same_previous(self) -> 'Individual':
    #     new_individual = Individual(self.neural_network_params, self.environment_class, self.environments_kwargs, self.parent_individual)
    #     new_individual.neural_network.set_parameters(self.neural_network.get_parameters())
    #     new_individual.fitness = self.fitness
    #     new_individual.is_fitness_calculated = self.is_fitness_calculated
    #     new_individual.params_to_look_at = self.params_to_look_at
    #     return new_individual

    # def __diff_evolution_iteration(self,
    #                                self_params: Dict[str, Any],
    #                                 base_params: Dict[str, Any],
    #                                 individual_1_params: Dict[str, Any],
    #                                 individual_2_params: Dict[str, Any],
    #                                 cross_prob: float,
    #                                 diff_weight: float) -> None:
    #     """
    #     Performs Differential Evolution on np.array, iterates fthrwo params
    #     :param self_params:
    #     :param base_params:
    #     :param individual_1_params:
    #     :param individual_2_params:
    #     :param cross_prob:
    #     :param diff_weight:
    #     :return:
    #     """
    #     for key, value in self_params.items():
    #         if isinstance(value, dict):
    #             self.__diff_evolution_iteration(value, base_params[key], individual_1_params[key], individual_2_params[key], cross_prob, diff_weight)
    #         elif isinstance(value, np.ndarray):
    #             cross_mask = np.random.rand(*value.shape) < cross_prob
    #             self_params[key][cross_mask] = base_params[key][cross_mask] + diff_weight * (individual_1_params[key][cross_mask] - individual_2_params[key][cross_mask])


    # def __cross_mean_policy(self, params_to_change: Dict[str, Any], other_params: Dict[str, Any]) -> None:
    #     """
    #     Changes mean policy dictionary parameters, params_to_change = (params_to_change + other_params) / 2
    #     :param params_to_change: current main policy part of params
    #     :param other_params: params from other individual
    #     :param key_list: list of keys to get to the parameter
    #     """
    #     for key, value in params_to_change.items():
    #         if isinstance(value, dict):
    #             self.__cross_mean_policy(value, other_params[key])
    #         elif isinstance(value, np.ndarray):
    #             params_to_change[key] = (value + other_params[key]) / 2



    # def _actualise_policy_dict_params(self, main_policy: Dict[str, Any], other_policies: List[Dict[str, Any]], change_amount: np.ndarray, key_list: List[str]) -> None:
    #     """
    #     Actualises policy dictionary parameters
    #     :param main_policy: current main policy part of params
    #     :param other_policies: list of other policies
    #     :param change_amount: fitnesses * multiply_factor
    #     :param key_list: list of keys to get to the parameter
    #     :return:
    #     """
    #     for key, value in main_policy.items():
    #         key_list.append(key)
    #         if isinstance(value, dict):
    #             self._actualise_policy_dict_params(value, other_policies, change_amount, key_list)
    #         elif isinstance(value, np.ndarray):
    #             param_array_double = value.astype(np.float64)
    #             for i in range(len(change_amount)):
    #                 param_array_double += change_amount[i] * self._get_element_from_dict(other_policies[i], key_list)
    #             main_policy[key] = param_array_double.astype(np.float32)
    #         key_list.pop()
    #
    # def _get_element_from_dict(self, dictionary: Dict[str, Any], key_list: List[str]) -> Any:
    #     """
    #     Gets element from dictionary by key list
    #     :param dictionary:
    #     :param key_list:
    #     :return:
    #     """
    #     for key in key_list:
    #         dictionary = dictionary[key]
    #     return dictionary

    # def _cross_scattered_policy(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> None:
    #     """
    #     changes params_to_change so that on random places it swapes params1 and params2
    #     :param params1: first policy part of params
    #     :param params2: second policy part of params
    #     """
    #     for key, value in params1.items():
    #         if isinstance(value, dict):
    #             self._cross_scattered_policy(value, params2[key])
    #         elif isinstance(value, np.ndarray):
    #             mask = np.random.randint(0, 2, value.shape, dtype=np.bool_)
    #             params_2_mask_copy = np.array(params2[key][mask])
    #             params2[key][mask] = value[mask]
    #             params1[key][mask] = params_2_mask_copy


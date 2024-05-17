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
# Because of mutation_factor change, currently only works normal mutation!
# L1 and L2 factors are implemented only for mutation_threshold = None
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

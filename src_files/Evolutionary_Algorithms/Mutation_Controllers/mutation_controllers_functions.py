from abc import ABC, abstractmethod
from typing import Any, Type


class Abstract_Mutation_Controller(ABC):
    @abstractmethod
    def mutate(self, params: dict[str, Any]) -> int:
        """
        Mutates the given parameters inplace, returns mutation id
        :param params:
        :return:
        """

    @abstractmethod
    def mutation_better_than_parent(self, id: int, parent_fitness: float, child_fitness: float) -> None:
        """
        Tells the mutation controller that the mutation with the given id is better than the parent
        """

    @abstractmethod
    def commit_iteration(self) -> None:
        """
        Tells the mutation controller that the iteration is over
        """


def get_mutation_controller_by_name(name: str) -> Type[Abstract_Mutation_Controller]:
    """
    Returns the mutation controller class by name
    :param name:
    :return:
    """
    if name == "SHADE":
        from src_files.Evolutionary_Algorithms.Mutation_Controllers.SHADE import SHADE
        return SHADE
    else:
        raise ValueError("Unknown mutation controller name: " + name)
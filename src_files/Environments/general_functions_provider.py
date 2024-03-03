from typing import Optional, Type

from src_files.Environments.Abstract_Environment.Abstract_Environment import Abstract_Environment


def get_environment_class(environment_name: str) -> Optional[Type[Abstract_Environment]]:
    """
    Returns environment class by name
    :param environment_name: available: "Basic_Car_Environment"
    :return:
    """
    if environment_name == "Basic_Car_Environment":
        from src_files.Environments.Basic_Car_Environment.Basic_Car_Environment import Basic_Car_Environment
        return Basic_Car_Environment
    else:
        raise ValueError(f"Unknown environment name {environment_name}")
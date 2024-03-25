from typing import Dict, Any

import numpy as np


cdef class Abstract_Parametrized_Layer(Abstract_Layer):
    def generate_parameters(self) -> None:
        """
        Generates new parameters for layer
        """
        raise NotImplementedError

    def get_parameters(self) -> Dict[str, Any]:
        """
        Returns parameters of layer
        :return: Dict[str, Dict[str, ...] or np.ndarray]
        """
        raise NotImplementedError

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Sets parameters of layer
        :param parameters: Dict[str, Dict[str, ...] or np.ndarray]
        :return: None
        """
        raise NotImplementedError

    def get_safe_mutation(self) -> Dict[str, Any]:
        """
        Returns value for parameters that determine safe mutation size
        """
        raise NotImplementedError


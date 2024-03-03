import json
import pickle
from typing import Optional, Dict, Any

import cython
import numpy as np

from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Dense_Layer.Dense_Layer import Dense_Layer
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Parameter_Generator import Xavier_Distribution_Generator, \
    He_Distribution_Generator
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Sequence_Layers.Sequence_Layers import Sequence_Layers
from src_files.Neural_Network.Raw_Numpy.general_functions_provider import get_activation_class
from src_files.MyMath.cython_debug_helper import cython_debug_call

cdef class Normal_model:

    @classmethod
    def __create_empty_model(cls: 'Normal_model') -> 'Normal_model':
        return cls

    def __init__(self, input_normal_size: int, out_actions_number: int = 3, normal_hidden_layers: int = 1,
                 normal_hidden_neurons: int = 64, normal_activation_function: str = "relu",
                 last_activation_function: str = "none") -> None:
        """
        Create a new model with the given parameters
        :param input_normal_size: size of the input for the normal part
        :param out_actions_number: number of output actions
        :param normal_hidden_layers: number of hidden layers in the normal part
        :param normal_hidden_neurons: number of neurons in the hidden layers in the normal part
        :param normal_activation_function: activation function for the normal part: "relu", "tanh", "sigmoid", "softmax", "none"
        :param last_activation_function: activation function for the last layer in the normal part: "relu", "tanh", "sigmoid", "softmax", "none"
        """
        self.normal_input_size = input_normal_size

        layers_counter: int = 0
        last_sequence_layer: Optional[Sequence_Layers] = None
        last_activation_class = get_activation_class(last_activation_function)
        add_hidden_activation = False

        if last_activation_class is not None:
            last_sequence_layer = Sequence_Layers(last_activation_class(), last_sequence_layer, layers_counter)
            layers_counter += 1

        current_out_neurons = out_actions_number
        current_in_neurons = normal_hidden_neurons

        activation_class = get_activation_class(normal_activation_function)
        if normal_activation_function == "relu" or normal_activation_function == "none":
            parameter_generator = He_Distribution_Generator()
        else:
            parameter_generator = Xavier_Distribution_Generator()

        for i in range(normal_hidden_layers + 1):
            if activation_class is not None and add_hidden_activation:
                last_sequence_layer = Sequence_Layers(activation_class(), last_sequence_layer, layers_counter)
                layers_counter += 1
            add_hidden_activation = True

            if i == normal_hidden_layers:
                current_in_neurons = input_normal_size
            # cython_debug_call(
            #     {
            #         "activation_class": activation_class,
            #         "add_hidden_activation": add_hidden_activation,
            #         "counter": layers_counter,
            #         "in neurons": current_in_neurons,
            #         "out neurons": current_out_neurons,
            #         "i": i,
            #         "normal_hidden neurons": normal_hidden_layers,
            #         "input_normal_size": input_normal_size,
            #     },
            #     "layer creation"
            # )
            last_sequence_layer = Sequence_Layers(Dense_Layer(current_in_neurons, current_out_neurons, parameter_generator), last_sequence_layer, layers_counter)
            current_out_neurons = normal_hidden_neurons
            layers_counter += 1

        self.normal_part = last_sequence_layer

    def copy(self) -> 'Normal_model':
        """Create a copy of the model, can be used in multiprocessing."""
        new_model = Normal_model.__create_empty_model()
        new_model.normal_part = self.normal_part.copy()
        return new_model

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set the parameters of the model
        :param parameters:
        :return:
        """
        self.normal_part.set_parameters(parameters)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters of the model
        :return:
        """
        return self.normal_part.get_parameters()

    def save_parameters(self, file_path: str) -> bool:
        """
        Save the parameters of the model to a file
        :param file_path:
        :return:
        """
        params = self.get_parameters()
        try:
            with open(file_path, "wb") as file:
                pickle.dump(params, file)
        except Exception as e:
            return False

        no_numpy_dict = self.__denumpy_dictionary(params)

        if file_path.endswith(".pkl"):
            file_path_json = file_path.replace(".pkl", ".json")
        else:
            file_path_json = file_path + ".json"

        try:
            with open(file_path_json, "w") as file:
                json.dump(no_numpy_dict, file, indent=4)

        except Exception as e:
            return False

        return True

    def load_parameters(self, file_path: str) -> bool:
        """
        Load the parameters of the model from a file, loads only .pkl files
        :param file_path:
        :return:
        """
        try:
            with open(file_path, "rb") as file:
                params = pickle.load(file)
        except Exception as e:
            return False

        self.set_parameters(params)
        return True

    def __denumpy_dictionary(self, dictionary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the numpy arrays in the dictionary to lists
        :param dictionary:
        :return:
        """
        new_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                new_dict[key] = value.tolist()
            else:
                new_dict[key] = self.__denumpy_dictionary(value) if isinstance(value, dict) else value
        return new_dict

    def create_new_model(self) -> None:
        """
        Creates new model from model architecture and model architecture kwargs, should be called in constructor nad to reset model
        :return:
        """
        self.normal_part.generate_parameters()

    cdef int get_normal_input_size(self) noexcept nogil:
        """
        Get the size of the input for the normal part
        """
        return self.normal_input_size


    def p_forward_pass(self, normal_input: np.ndarray) -> np.ndarray:
        """
        Forward pass for the normal part
        :param normal_input:
        :return:
        """
        return np.array(self.forward_pass(np.array(normal_input, dtype=np.float32)), dtype=np.float32)

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef float[:, ::1] forward_pass(self, float[:, ::1] normal_input) noexcept nogil:
        return self.normal_part.forward(normal_input)

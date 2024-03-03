from typing import Dict

import numpy as np
cimport cython

from src_files.MyMath.cython_debug_helper import cython_debug_call
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers import Parameter_Generator
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Abstract_Parametrized_Layer.Abstract_Parametrized_Layer cimport Abstract_Parametrized_Layer

# I used float_32 here, cause I think I doesnt need more precision and chatGpt told me that it is faster than float_64



cdef class Dense_Layer(Abstract_Parametrized_Layer):
    cdef float[:, ::1] weights  # neurons are columns! rows are weights for specific input!
    cdef float[::1] biases  # biases for each neuron, actually these attributes are private
    cdef float[:, ::1] output  # output of the layer
    cdef int input_size
    cdef int output_size
    cdef object parameters_generator

    def __init__(self, input_size: int, output_size: int, parameters_generator: Parameter_Generator):
        # only python attributes
        self.input_size = input_size
        self.output_size = output_size
        self.parameters_generator = parameters_generator

        # c attributes
        self.generate_parameters()
        self.output = np.zeros([1, self.output_size], dtype=np.float32)

    def copy(self) -> 'Dense_Layer':
        """
        returns a copy of the layer
        :return:
        """
        new_layer = Dense_Layer(self.input_size, self.output_size, self.parameters_generator)
        new_layer.set_parameters(self.get_parameters())
        return new_layer

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        returns the parameters of the layer
        :return:
        """
        return {"weights": np.array(self.weights, copy=True), "biases": np.array(self.biases, copy=True)}

    def set_parameters(self, parameters: Dict[str, np.ndarray]) -> None:
        """
        sets the parameters of the layer
        :param parameters:
        :return:
        """
        if "weights" in parameters:
            self.weights = parameters["weights"]

        if "biases" in parameters:
            self.biases = parameters["biases"]

    def generate_parameters(self) -> None:
        """
        generates new weights and biases from the parameters_generator
        :return:
        """
        self.weights = self.parameters_generator.generate_weights([self.input_size, self.output_size])
        self.biases = self.parameters_generator.generate_biases([self.output_size])

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef inline float[:, ::1] forward(self, float[:, ::1] inputs) noexcept nogil:
        """
        :param inputs: 
        :return: 
        """
        if inputs.shape[0] > self.output.shape[0]:
            with gil:
                self.output = np.zeros([inputs.shape[0], self.output_size], dtype=np.float32)

        cdef float[:, ::1] weights_here = self.weights
        cdef float[::1] biases_here = self.biases
        cdef float[:, ::1] output_here = self.output[:inputs.shape[0]]
        cdef int batch_size = inputs.shape[0]
        cdef int inputs_size = weights_here.shape[0]
        cdef int output_size = output_here.shape[1]
        cdef int i, j, k

        for i in range(batch_size):
            for j in range(output_size):
                output_here[i, j] = biases_here[j]
                for k in range(inputs_size):
                    output_here[i, j] += inputs[i, k] * weights_here[k, j]
        # with gil:
        #     cython_debug_call(
        #         {
        #             "weights": np.array(weights_here),
        #             "biases": np.array(biases_here),
        #             "inputs": np.array(inputs),
        #             "output": np.array(output_here),
        #             "batch_size": batch_size,
        #             "inputs_size": inputs_size,
        #             "output_size": output_size,
        #         }
        #     )

        return output_here

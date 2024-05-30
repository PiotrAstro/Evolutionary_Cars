from typing import Dict

import numpy as np
cimport cython
from src_files.MyMath.MyMath cimport float_abs
from libc.math cimport sqrt
from src_files.MyMath.cython_debug_helper import cython_debug_call
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers import Parameter_Generator
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Abstract_Parametrized_Layer.Abstract_Parametrized_Layer cimport Abstract_Parametrized_Layer

# I used float_32 here, cause I think I doesnt need more precision and chatGpt told me that it is faster than float_64



cdef class Dense_Layer(Abstract_Parametrized_Layer):
    cdef float[:, ::1] weights  # neurons are columns! rows are weights for specific input!
    cdef float[::1] biases  # biases for each neuron, actually these attributes are private
    cdef float[:, ::1] output  # output of the layer
    cdef float[:, ::1] inputs_for_gradient  # inputs for the gradient calculation
    # cdef float[:, ::1] safe_mutation_abs_gradient_weights_sum_cache  # gradient information for the weights
    # cdef float[::1] safe_mutation_abs_gradient_biases_sum_cache  # gradient information for the biases
    cdef float[:, ::1] safe_mutation_weights_cache  # cache for the safe mutation gradients
    cdef float[::1] safe_mutation_biases_cache  # sum of the gradients for the weights
    cdef float[:, ::1] grad_weights_cache  # cache for the gradients
    cdef float[::1] grad_biases_cache  # cache for the gradients
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
        # self.safe_mutation_abs_gradient_weights_sum_cache = np.zeros_like(self.weights)
        # self.safe_mutation_abs_gradient_biases_sum_cache = np.zeros_like(self.biases)
        self.safe_mutation_weights_cache = np.zeros_like(self.weights)
        self.safe_mutation_biases_cache = np.zeros_like(self.biases)
        self.grad_weights_cache = np.zeros_like(self.weights)
        self.grad_biases_cache = np.zeros_like(self.biases)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def get_safe_mutation(self) -> Dict[str, np.ndarray]:
        """
        returns the gradients of the layer
        :return:
        """
        cdef int rows, cols, i, j
        cdef float[:, ::1] safe_mutation_weights_cache
        cdef float[::1] safe_mutation_biases_cache
        with nogil:
            rows = self.safe_mutation_weights_cache.shape[0]
            cols = self.safe_mutation_weights_cache.shape[1]
            safe_mutation_weights_cache = self.safe_mutation_weights_cache
            safe_mutation_biases_cache = self.safe_mutation_biases_cache

            for i in range(rows):
                for j in range(cols):
                    safe_mutation_weights_cache[i, j] = sqrt(safe_mutation_weights_cache[i, j])
            for i in range(cols):
                safe_mutation_biases_cache[i] = sqrt(safe_mutation_biases_cache[i])

        result = {"weights": np.array(self.safe_mutation_weights_cache, copy=True), "biases": np.array(self.safe_mutation_biases_cache, copy=True)}

        with nogil:
            for i in range(rows):
                for j in range(cols):
                    safe_mutation_weights_cache[i, j] = 0
            for i in range(cols):
                safe_mutation_biases_cache[i] = 0
        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef inline float[:, ::1] backward(self, float[:, ::1] grad) noexcept nogil:
        """
        :param grad: shape (batch_size, num_classes)
        :return: 
        """
        cdef int i, j, k
        cdef int rows = grad.shape[0]
        cdef int out_cols = grad.shape[1]
        cdef int in_cols = self.weights.shape[0]
        cdef float[:, ::1] prev_input = self.previous_inputs
        cdef float[:, ::1] prev_output = self.previous_outputs
        cdef float[:, ::1] grads_inputs = self.grads_inputs
        cdef float[:, ::1] weights = self.weights
        cdef float[::1] biases = self.biases
        # cdef float[:, ::1] safe_mutation_abs_gradient_weights_sum_cache = self.safe_mutation_abs_gradient_weights_sum_cache
        # cdef float[::1] safe_mutation_abs_gradient_biases_sum_cache = self.safe_mutation_abs_gradient_biases_sum_cache
        # cdef float weight_grad_abs_sum_tmp
        # cdef float bias_grad_abs_sum_tmp
        # cdef float[:, ::1] safe_mutation_weights_cache = self.safe_mutation_weights_cache
        # cdef float[::1] safe_mutation_biases_cache = self.safe_mutation_biases_cache
        cdef float[:, ::1] grad_weights_cache = self.grad_weights_cache
        cdef float[::1] grad_biases_cache = self.grad_biases_cache


        for i in range(out_cols):
            bias_grad_abs_sum_tmp = 0
            grad_biases_cache[i] = 0
            for j in range(in_cols):
                weight_grad_abs_sum_tmp = 0
                grad_weights_cache[j, i] = 0
                for k in range(rows):
                    grad_weights_cache[j, i] += grad[k, i] * prev_input[k, j]
                    grad_biases_cache[i] += grad[k, i]
            #         weight_grad_abs_sum_tmp += float_abs(grad[k, i] * prev_input[k, j])
            #         bias_grad_abs_sum_tmp += float_abs(grad[k, i])
            #     safe_mutation_weights_cache[j, i] += (weight_grad_abs_sum_tmp / rows) ** 2
            # safe_mutation_biases_cache[i] += (bias_grad_abs_sum_tmp / rows) ** 2

        for k in range(rows):
            for j in range(in_cols):
                grads_inputs[k, j] = 0
                for i in range(out_cols):
                    grads_inputs[k, j] += grad[k, i] * weights[j, i]

        # with gil:
        #     cython_debug_call({
        #         "grads_inputs": np.array(grads_inputs),
        #         "grad": np.array(grad),
        #         "prev_outputs": np.array(self.previous_outputs),
        #         "prev_inputs": np.array(self.previous_inputs),
        #         "safe_mutation_weights_cache": np.array(self.safe_mutation_weights_cache),
        #         "safe_mutation_biases_cache": np.array(self.safe_mutation_biases_cache),
        #     },
        #     "Dense_Layer_backward"
        #     )

        return grads_inputs

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef int SGD(self, float learning_rate) noexcept nogil:
        """
        :param learning_rate: 
        :return: 
        """
        cdef float[:, ::1] weights = self.weights
        cdef float[::1] biases = self.biases
        cdef float[:, ::1] grad_weights_cache = self.grad_weights_cache
        cdef float[::1] grad_biases_cache = self.grad_biases_cache
        cdef int i, j
        cdef int rows = weights.shape[0]
        cdef int cols = weights.shape[1]

        for i in range(rows):
            for j in range(cols):
                weights[i, j] -= learning_rate * grad_weights_cache[i, j]

        for i in range(cols):
            biases[i] -= learning_rate * grad_biases_cache[i]

        # with gil:
        #     cython_debug_call({
        #         "weights": np.array(weights),
        #         "biases": np.array(biases),
        #         "grad_weights_cache": np.array(grad_weights_cache),
        #         "grad_biases_cache": np.array(grad_biases_cache),
        #     },
        #     "Dense_Layer_SGD"
        #     )
        return 0

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef inline float[:, ::1] forward(self, float[:, ::1] inputs) noexcept nogil:
        if inputs.shape[0] > self.output.shape[0]:
            with gil:
                self.output = np.empty([inputs.shape[0], self.output_size], dtype=np.float32)

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

import numpy as np
from libc.math cimport exp
from src_files.MyMath.cython_debug_helper import cython_debug_call
import cython

from src_files.MyMath.MyMath cimport lookup_exp
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Abstract_Layer.Abstract_Layer cimport Abstract_Layer



cdef class Softmax_Activation(Abstract_Layer):
    def __init__(self):
        pass

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef inline float[:, ::1] forward(self, float[:, ::1] inputs) noexcept nogil:
        """
        :param inputs: shape (batch_size, num_classes)
        :return: 
        """
        cdef int rows = inputs.shape[0]
        cdef int cols = inputs.shape[1]
        cdef double row_sum
        cdef double exp_tmp
        cdef float max_value = -100000000

        for i in range(rows):
            row_sum = 0
            for j in range(cols):
                if inputs[i, j] > max_value:
                    max_value = inputs[i, j]
            for j in range(cols):
                exp_tmp = lookup_exp(inputs[i, j] - max_value)
                row_sum += exp_tmp
                inputs[i, j] = exp_tmp
            for j in range(cols):
                inputs[i, j] = inputs[i, j] / row_sum
        return inputs

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef inline float[:, ::1] backward(self, float[:, ::1] grad) noexcept nogil:
        """
        Compute the gradient of the loss with respect to the inputs of the softmax layer.

        :param dvalues: The gradient of the loss with respect to the outputs of the softmax layer.
                        shape (batch_size, num_classes)
        :return: The gradient of the loss with respect to the inputs of the softmax layer.
                 shape (batch_size, num_classes)
        """
        cdef int i, j, k
        cdef int rows = grad.shape[0]
        cdef int cols = grad.shape[1]
        cdef float[:, ::1] prev_outputs = self.previous_outputs
        cdef float[:, ::1] grads_inputs = self.grads_inputs

        for k in range(rows):
            for i in range(cols):
                grads_inputs[k, i] = 0
                for j in range(cols):
                    if i == j:
                        grads_inputs[k, i] += prev_outputs[k, i] * (1 - prev_outputs[k, i]) * grad[k, j]
                    else:
                        grads_inputs[k, i] += -prev_outputs[k, i] * prev_outputs[k, j] * grad[k, j]

        # with gil:
        #     cython_debug_call({
        #         "grads_inputs": np.array(grads_inputs),
        #         "grad": np.array(grad),
        #         "prev_outputs": np.array(self.previous_outputs),
        #         "prev_inputs": np.array(self.previous_inputs),
        #     }, "Softmax_Activation_backward")

        return grads_inputs


    def copy(self) -> 'Softmax_Activation':
        return Softmax_Activation()
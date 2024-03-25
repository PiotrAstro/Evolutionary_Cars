import numpy as np
from libc.math cimport exp
from src_files.MyMath.cython_debug_helper import cython_debug_call
import cython

from src_files.MyMath.MyMath cimport sigmoid
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Abstract_Layer.Abstract_Layer cimport Abstract_Layer


cdef class Sigmoid_Activation(Abstract_Layer):
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
        cdef int i, j

        for i in range(rows):
            for j in range(cols):
                inputs[i, j] = sigmoid(inputs[i, j])
        return inputs

    def copy(self) -> 'Sigmoid_Activation':
        return Sigmoid_Activation()

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
        cdef float[:, ::1] grads_inputs = self.grads_inputs
        cdef float[:, ::1] prev_outputs = self.previous_outputs

        for k in range(rows):
            for i in range(cols):
                grads_inputs[k, i] = grad[k, i] * prev_outputs[k, i] * (1 - prev_outputs[k, i])

        # with gil:
        #     cython_debug_call({
        #         "grads_inputs": np.array(grads_inputs),
        #         "grad": np.array(grad),
        #         "prev_outputs": np.array(self.previous_outputs),
        #         "prev_inputs": np.array(self.previous_inputs),
        #     },
        #     "Sigmoid_backward")

        return grads_inputs
import cython
import numpy as np
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Abstract_Layer.Abstract_Layer cimport Abstract_Layer
from src_files.MyMath.cython_debug_helper import cython_debug_call

cdef class Relu_Activation(Abstract_Layer):
    def __init__(self):
        pass

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef inline float[:, ::1] forward(self, float[:, ::1] inputs) noexcept nogil:
        cdef int rows = inputs.shape[0]
        cdef int cols = inputs.shape[1]
        cdef int i, j

        for i in range(rows):
            for j in range(cols):
                if inputs[i, j] < 0:
                    inputs[i, j] = 0
        return inputs

    def copy(self) -> 'Relu_Activation':
        return Relu_Activation()

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
        cdef int cols = grad.shape[1]
        cdef float[:, ::1] prev_inputs = self.previous_inputs
        cdef float[:, ::1] grads_inputs = self.grads_inputs

        for k in range(rows):
            for i in range(cols):
                if prev_inputs[k, i] <= 0:
                    grads_inputs[k, i] = 0
                else:
                    grads_inputs[k, i] = grad[k, i]

        # with gil:
        #     cython_debug_call({
        #         "grads_inputs": np.array(grads_inputs),
        #         "grad": np.array(grad),
        #         "prev_outputs": np.array(self.previous_outputs),
        #         "prev_inputs": np.array(self.previous_inputs),
        #     },
        #         "Relu_backward")

        return grads_inputs
from libc.math cimport exp

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
        cdef float max_value = -100000000

        for i in range(rows):
            row_sum = 0
            for j in range(cols):
                if inputs[i, j] > max_value:
                    max_value = inputs[i, j]
            for j in range(cols):
                inputs[i, j] = lookup_exp(inputs[i, j] - max_value)
                row_sum += inputs[i, j]
            for j in range(cols):
                inputs[i, j] = inputs[i, j] / row_sum

    def copy(self) -> 'Softmax_Activation':
        return Softmax_Activation()
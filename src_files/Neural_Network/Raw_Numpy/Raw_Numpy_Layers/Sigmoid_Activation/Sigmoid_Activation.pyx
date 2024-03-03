from libc.math cimport exp

import cython

from src_files.MyMath.MyMath cimport lookup_exp
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
                inputs[i, j] = 1.0 / (1.0 + lookup_exp(-inputs[i, j]))
        return inputs

    def copy(self) -> 'Sigmoid_Activation':
        return Sigmoid_Activation()


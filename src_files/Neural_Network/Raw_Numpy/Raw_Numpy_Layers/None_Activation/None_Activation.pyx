from libc.math cimport exp

import cython

from src_files.MyMath.MyMath cimport sigmoid
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Abstract_Layer.Abstract_Layer cimport Abstract_Layer


cdef class None_Activation(Abstract_Layer):
    def __init__(self):
        pass

    cdef inline float[:, ::1] forward(self, float[:, ::1] inputs) noexcept nogil:
        """
        :param inputs: shape (batch_size, num_classes)
        :return: 
        """
        return inputs

    def copy(self) -> 'None_Activation':
        return None_Activation()


    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef inline float[:, ::1] backward(self, float[:, ::1] grad) noexcept nogil:
        """
        :param grad: shape (batch_size, num_classes)
        :return: 
        """
        cdef float[:, ::1] grads_here = self.grads_inputs
        cdef int rows = grad.shape[0]
        cdef int cols = grad.shape[1]
        for i in range(rows):
            for j in range(cols):
                grads_here[i, j] = grad[i, j]

        return grads_here
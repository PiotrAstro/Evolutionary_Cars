import cython
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Abstract_Layer.Abstract_Layer cimport Abstract_Layer

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

from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Abstract_Parametrized_Layer.Abstract_Parametrized_Layer cimport Abstract_Parametrized_Layer
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Abstract_Layer.Abstract_Layer cimport Abstract_Layer


cdef class Sequence_Layers(Abstract_Parametrized_Layer):
    cdef Sequence_Layers next_one
    cdef Abstract_Layer layer
    cdef int self_number
    cdef object layer_name  # python string
    cdef float[:, ::1] forward(self, float[:, ::1] inputs) noexcept nogil
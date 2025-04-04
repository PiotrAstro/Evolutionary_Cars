from typing import Dict, Any

from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Abstract_Layer.Abstract_Layer cimport Abstract_Layer

cdef class Abstract_Parametrized_Layer(Abstract_Layer):
    cdef float dropout_rate
    cdef unsigned char [:, ::1] dropout_mask
    pass

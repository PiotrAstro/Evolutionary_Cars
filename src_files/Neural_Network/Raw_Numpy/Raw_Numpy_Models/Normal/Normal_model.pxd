from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Sequence_Layers.Sequence_Layers cimport Sequence_Layers

cdef class Normal_model:
    cdef Sequence_Layers normal_part

    cdef int normal_input_size

    cdef float[:, ::1] forward_pass(self, float[:, ::1] normal_input) noexcept nogil

    cdef int get_normal_input_size(self) noexcept nogil

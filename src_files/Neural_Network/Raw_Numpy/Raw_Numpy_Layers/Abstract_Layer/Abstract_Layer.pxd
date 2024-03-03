

cdef class Abstract_Layer:
    cdef float[:, ::1] forward(self, float[:, ::1] inputs) noexcept nogil


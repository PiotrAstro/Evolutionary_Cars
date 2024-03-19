# cdef object real_t_numpy
cdef class Abstract_Environment:
    cdef int reset(self) noexcept nogil
    cdef float[::1] get_state(self) noexcept nogil
    cdef double react(self, float[::1] outputs) noexcept nogil
    cdef bint is_alive(self) noexcept nogil
    cdef int get_state_length(self) noexcept nogil


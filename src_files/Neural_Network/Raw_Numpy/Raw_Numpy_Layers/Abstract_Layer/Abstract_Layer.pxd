

cdef class Abstract_Layer:
    cdef float[:, ::1] inputs_copy

    # for gradient return, every layer should use this one!
    cdef float[:, ::1] grads_inputs
    cdef float[:, ::1] previous_inputs
    # carefull! If I calculate this, I cant change output values, cause it is used for backpropagation
    cdef float[:, ::1] previous_outputs  # currently it just takes same memory of forward output, consider it, as sometimes I might have to copy it
    cdef float[:, ::1] forward(self, float[:, ::1] inputs) noexcept nogil
    cdef float[:, ::1] forward_grad(self, float[:, ::1] inputs) noexcept nogil
    cdef float[:, ::1] backward(self, float[:, ::1] grad) noexcept nogil
    cdef int SGD(self, float learning_rate) noexcept nogil



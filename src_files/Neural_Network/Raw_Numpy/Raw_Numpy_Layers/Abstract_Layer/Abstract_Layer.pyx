
cdef class Abstract_Layer:
    def copy(self) -> 'Abstract_Layer':
        """
        Returns copy of layer
        :return:
        """
        pass

    cdef float[:, ::1] forward(self, float[:, ::1] inputs) noexcept nogil:
        """
        Calls layer
        :param inputs: np.ndarray
        """
        with gil:
            raise NotImplementedError
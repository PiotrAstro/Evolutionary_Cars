import numpy as np

real_t_numpy = np.float32

cdef class Abstract_Environment:
    def p_reset(self) -> None:
        """
        Reset the environment to its initial state
        :return:
        """
        self.reset()


    def p_get_state(self):
        """
        Get the current state of the environment as np.ndarray
        :return:
        """
        return np.array(self.get_state(), dtype=np.float32)

    def p_react(self, outputs: np.ndarray) -> float:
        """
        React to the outputs of the agent, return the reward
        :param outputs:
        :return:
        """
        return self.react(np.array(outputs, dtype=np.float32))

    def p_is_alive(self) -> bool:
        """
        Check if the environment is still alive
        :return: bool
        """
        return self.is_alive()

    def p_get_state_length(self) -> int:
        """
        Get the length of the state vector
        :return:
        """
        return self.get_state_length()




    # Abstract methods
    cdef int reset(self) noexcept nogil:
        with gil:
            raise NotImplementedError("Abstract method")

    cdef real_t[::1] get_state(self) noexcept nogil:
        with gil:
            raise NotImplementedError("Abstract method")

    cdef double react(self, real_t[::1] outputs) noexcept nogil:
        with gil:
            raise NotImplementedError("Abstract method")

    cdef bint is_alive(self) noexcept nogil:
        with gil:
            raise NotImplementedError("Abstract method")

    cdef int get_state_length(self) noexcept nogil:
        with gil:
            raise NotImplementedError("Abstract method")
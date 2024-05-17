import cython
import numpy as np
from src_files.MyMath.cython_debug_helper import cython_debug_call

cdef class Abstract_Layer:

    def __cinit__(self):
        self.previous_inputs = None
        self.previous_outputs = None
        self.grads_inputs = None
        self.inputs_copy = None

    def copy(self) -> 'Abstract_Layer':
        """
        Returns copy of layer
        :return:
        """
        pass

    cdef int SGD(self, float learning_rate) noexcept nogil:
        """
        Stochastic Gradient Descent
        :param learning_rate: float
        :return: int
        """
        pass


    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef float[:, ::1] forward_grad(self, float[:, ::1] inputs) noexcept nogil:
        """
        performs normal grad, but saves inputs for backpropagation
        """
        cdef float[:, ::1] inputs_copy
        cdef int rows
        cdef int cols
        cdef int i, j

        if self.inputs_copy is None or self.inputs_copy.shape[0] != inputs.shape[0]:
            with gil:
                self.inputs_copy = np.array(inputs, dtype=np.float32, copy=True)
                self.grads_inputs = np.empty_like(inputs, dtype=np.float32)
            inputs_copy = self.inputs_copy
        else:
            inputs_copy = self.inputs_copy
            rows = inputs.shape[0]
            cols = inputs.shape[1]
            for i in range(rows):
                for j in range(cols):
                    inputs_copy[i, j] = inputs[i, j]
        self.previous_inputs = inputs
        self.previous_outputs = self.forward(inputs_copy)
        # with gil:
        #     cython_debug_call(
        #         {
        #             "previous_inputs": np.array(self.previous_inputs),
        #             "previous_outputs": np.array(self.previous_outputs),
        #             "grads_inputs": np.array(self.grads_inputs),
        #         }
        #     )

        return self.previous_outputs

    cdef float[:, ::1] forward(self, float[:, ::1] inputs) noexcept nogil:
        """
        Calls layer
        :param inputs: np.ndarray
        """
        with gil:
            raise NotImplementedError

    cdef float[:, ::1] backward(self, float[:, ::1] grad) noexcept nogil:
        """
        Calls layer
        :param grad: np.ndarray
        """
        with gil:
            raise NotImplementedError
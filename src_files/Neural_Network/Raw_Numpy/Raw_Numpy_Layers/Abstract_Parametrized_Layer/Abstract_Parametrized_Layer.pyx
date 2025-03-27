from typing import Dict, Any
import cython
import numpy as np
from libc.stdlib cimport rand, RAND_MAX


cdef class Abstract_Parametrized_Layer(Abstract_Layer):

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef float[:, ::1] forward_grad(self, float[:, ::1] inputs) noexcept nogil:
        """
        performs normal grad, but saves inputs for backpropagation
        """
        cdef float[:, ::1] inputs_copy
        cdef unsigned char [:, ::1] dropout_mask_here
        cdef int rows
        cdef int cols
        cdef int i, j

        if self.inputs_copy is None or self.inputs_copy.shape[0] != inputs.shape[0]:
            with gil:
                self.dropout_mask = np.empty_like(inputs, dtype=np.uint8)
                self.inputs_copy = np.empty_like(inputs, dtype=np.float32)
                self.grads_inputs = np.empty_like(inputs, dtype=np.float32)

        dropout_mask_here = self.dropout_mask
        inputs_copy = self.inputs_copy
        rows = inputs.shape[0]
        cols = inputs.shape[1]
        for i in range(rows):
            for j in range(cols):
                if rand() < self.dropout_rate * RAND_MAX:
                    inputs_copy[i, j] = 0
                    dropout_mask_here[i, j] = 0
                else:
                    inputs_copy[i, j] = inputs[i, j]
                    dropout_mask_here[i, j] = 1

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

    def generate_parameters(self) -> None:
        """
        Generates new parameters for layer
        """
        raise NotImplementedError

    def get_parameters(self) -> Dict[str, Any]:
        """
        Returns parameters of layer
        :return: Dict[str, Dict[str, ...] or np.ndarray]
        """
        raise NotImplementedError

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Sets parameters of layer
        :param parameters: Dict[str, Dict[str, ...] or np.ndarray]
        :return: None
        """
        raise NotImplementedError

    def get_safe_mutation(self) -> Dict[str, Any]:
        """
        Returns value for parameters that determine safe mutation size
        """
        raise NotImplementedError



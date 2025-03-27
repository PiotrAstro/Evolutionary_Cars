import math

import cython

from src_files.Environments.Abstract_Environment.Abstract_Environment cimport Abstract_Environment
from src_files.MyMath.cython_debug_helper import cython_debug_call

from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Models.Normal.Normal_model cimport Normal_model
from typing import List

import numpy as np



cdef class Abstract_Environment_Iterator:
    cdef Abstract_Environment_Iterator next_it
    cdef Abstract_Environment self_environment
    cdef int number_of_environments
    cdef bint is_self_alive


    def __init__(self, environments_list: List[Abstract_Environment]):
        if len(environments_list) == 1:
            self.self_environment = environments_list[0]
            self.next_it = None
        else:
            self.self_environment = environments_list[0]
            self.next_it = Abstract_Environment_Iterator(environments_list[1:])
        self.number_of_environments = len(environments_list)
        self.is_self_alive = True

    def get_results(self, model: Normal_model) -> float:
        """
        This function returns sum of all results from all environments, it is for python use
        :return:
        """
        cdef Normal_model model_cython = model
        cdef float[:, ::1] input_states = np.zeros((self.number_of_environments, model_cython.get_normal_input_size()), dtype=np.float32)
        cdef float[:, ::1] outputs
        cdef int input_rows_number
        cdef double result = 0

        with nogil:
            self.iterate_reset_environments()

            while self.iterate_is_alive():
                input_rows_number = self.iterate_insert_state(input_states)
                outputs = model_cython.forward_pass(input_states[:input_rows_number])
                result += self.iterate_react(outputs)
        return result

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    # def get_results_with_input_output(self, model: Normal_model, max_length: int) -> tuple[float, np.ndarray, np.ndarray]:
    #     """
    #     This function returns sum of all results from all environments, it is for python use
    #     :return: sum of all results, input_states, outputs
    #     """
    #     cdef Normal_model model_cython = model
    #     cdef float[:, ::1] input_states_memory = np.empty((max_length, model_cython.get_normal_input_size()), dtype=np.float32)
    #     cdef float[:, ::1] outputs_memory = np.empty((max_length, model_cython.get_normal_output_size()), dtype=np.float32)
    #     cdef float[:, ::1] input_states = np.zeros((self.number_of_environments, model_cython.get_normal_input_size()), dtype=np.float32)
    #     cdef float[:, ::1] outputs
    #     cdef int memory_rows = 0
    #     cdef int max_length_here = max_length
    #     cdef int cols
    #     cdef int i, j
    #     cdef int input_rows_number
    #     cdef double result = 0
    #
    #     with nogil:
    #         self.iterate_reset_environments()
    #
    #         while self.iterate_is_alive():
    #             input_rows_number = self.iterate_insert_state(input_states)
    #
    #             if memory_rows + input_rows_number < max_length_here:
    #                 cols = input_states_memory.shape[1]
    #                 for i in range(input_rows_number):
    #                     for j in range(cols):
    #                         input_states_memory[memory_rows + i, j] = input_states[i, j]
    #
    #             outputs = model_cython.forward_pass(input_states[:input_rows_number])
    #
    #             if memory_rows + input_rows_number < max_length_here:
    #                 cols = outputs_memory.shape[1]
    #                 for i in range(input_rows_number):
    #                     for j in range(cols):
    #                         outputs_memory[memory_rows + i, j] = outputs[i, j]
    #             memory_rows += input_rows_number
    #
    #             result += self.iterate_react(outputs)
    #     # cython_debug_call({
    #     #     "input_states_memory": np.array(input_states_memory[:memory_rows], dtype=np.float32, copy=False),
    #     #     "outputs_memory": np.array(outputs_memory[:memory_rows], dtype=np.float32, copy=False),
    #     #     "result": result
    #     # },
    #     # "get_results_with_input_output"
    #     # )
    #     return result, np.array(input_states_memory[:memory_rows], dtype=np.float32, copy=False), np.array(outputs_memory[:memory_rows], dtype=np.float32, copy=False)


    cdef int iterate_insert_state(self, float[:, ::1] input_state) noexcept nogil:
        """
        This function inserts state to all environments after self
        :param input_state: 
        :return: number of valid inputs after self, it can be used to slice correctly
        """
        if self.is_self_alive:
            input_state[0] = self.self_environment.get_state()
            if self.next_it is not None:
                return 1 + self.next_it.iterate_insert_state(input_state[1:])
            return 1
        elif self.next_it is not None:
            return self.next_it.iterate_insert_state(input_state)
        return 0

    cdef int iterate_reset_environments(self) noexcept nogil:
        """
        This function resets all environments after self
        :return: 
        """
        self.is_self_alive = True
        self.self_environment.reset()
        if self.next_it is not None:
            return self.next_it.iterate_reset_environments()
        return 0

    cdef double iterate_react(self, float[:, ::1] output_values) noexcept nogil:
        """
        This function reacts all environments after self
        :param output_values: 
        :return: sum of all results from all environments
        """
        cdef double result = 0
        if self.is_self_alive:
            result = self.self_environment.react(output_values[0])
            if self.next_it is not None:
                result += self.next_it.iterate_react(output_values[1:])
            self.is_self_alive = self.self_environment.is_alive()
        elif self.next_it is not None:
            result += self.next_it.iterate_react(output_values)
        return result

    cdef bint iterate_is_alive(self) noexcept nogil:
        if self.is_self_alive:
            return True
        elif self.next_it is not None:
            return self.next_it.iterate_is_alive()
        return False
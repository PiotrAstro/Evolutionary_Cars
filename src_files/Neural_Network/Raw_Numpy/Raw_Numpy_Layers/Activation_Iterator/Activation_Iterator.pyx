from libc.math cimport exp

import cython
import numpy as np
from typing import Optional

from src_files.MyMath.MyMath cimport sigmoid
from src_files.MyMath.cython_debug_helper import cython_debug_call
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Abstract_Layer.Abstract_Layer cimport Abstract_Layer
from src_files.Neural_Network.Raw_Numpy.general_functions_provider import get_activation_class

cdef class Activation_Iterator:
    """
    It is Abstract Layer that can have different activations at different neurons, e.g. you can specify
    that neurons 0:3 have softmax, neurons 3:4 have tanh,
    You build it from back, so firstly you specify tanh, then softmax
    """

    cdef int num_activations
    cdef Abstract_Layer self_activation
    cdef Activation_Iterator next_activation

    def __init__(self, activation: Abstract_Layer, num_activations: int, next_activation: Optional[Activation_Iterator]):
        self.self_activation = activation
        self.num_activations = num_activations
        self.next_activation = next_activation

    cdef inline float[:, ::1] forward(self, float[:, ::1] inputs) noexcept nogil:
        # with gil:
        #     cython_debug_call({
        #         "inputs": np.array(inputs),
        #         "inputs for self": np.array(inputs[:, 0:self.num_activations]),
        #         "inputs for next": np.array(inputs[:, self.num_activations:]),
        #         "num self activations": self.num_activations,
        #     }, "Activation_Iterator forward")
        self.self_activation.forward(inputs[:, 0:self.num_activations])
        if self.next_activation is not None:
            self.next_activation.forward(inputs[:, self.num_activations:])
        return inputs

    cdef int backward(self, float[:, ::1] grad, float[:, ::1] grads_inputs, float[:, ::1] previous_outputs, float[:, ::1] previous_inputs) noexcept nogil:
        """
        :param grad: shape (batch_size, num_classes)
        :return: 
        """
        self.self_activation.grads_inputs = grads_inputs[:, 0:self.num_activations]
        self.self_activation.previous_outputs = previous_outputs[:, 0:self.num_activations]
        self.self_activation.previous_inputs = previous_inputs[:, 0:self.num_activations]
        self.self_activation.backward(grad[:, 0:self.num_activations])

        if self.next_activation is not None:
            self.next_activation.backward(
                grad[:, self.num_activations:],
                grads_inputs[:, self.num_activations:],
                previous_outputs[:, self.num_activations:],
                previous_inputs[:, self.num_activations:]
            )

        return 0

    def copy(self) -> 'Activation_Iterator':
        return Activation_Iterator(self.self_activation.copy(),
                                   self.num_activations,
                                   self.next_activation.copy() if self.next_activation is not None else None)
cdef class Activations_Iterator_Wrapper(Abstract_Layer):
    """
    It is Abstract Layer that can have different activations at different neurons, e.g. you can specify
    that neurons 0:3 have softmax, neurons 3:4 have tanh,
    You build it from back, so firstly you specify tanh, then softmax
    """
    cdef Activation_Iterator first_activation

    @classmethod
    def __create_empty_instance(cls):
        return cls

    def __init__(self, activations: list[tuple[str, int]]):
        self.first_activation = None
        for activation_name, activation_neurons in reversed(activations):
            self.first_activation = Activation_Iterator(get_activation_class(activation_name)(), activation_neurons, self.first_activation)

    cdef inline float[:, ::1] forward(self, float[:, ::1] inputs) noexcept nogil:
        return self.first_activation.forward(inputs)

    cdef float[:, ::1] backward(self, float[:, ::1] grad) noexcept nogil:
        self.first_activation.backward(grad, self.grads_inputs, self.previous_outputs, self.previous_inputs)
        return self.grads_inputs

    def copy(self) -> 'Activations_Iterator_Wrapper':
        new_instance = Activations_Iterator_Wrapper.__create_empty_instance()
        new_instance.first_activation = self.first_activation.copy()
        return new_instance
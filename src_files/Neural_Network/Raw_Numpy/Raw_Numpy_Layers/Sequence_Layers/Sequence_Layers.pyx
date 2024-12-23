from typing import Any, Dict, Optional

import numpy as np

from src_files.MyMath.cython_debug_helper import cython_debug_call

cdef class Sequence_Layers(Abstract_Parametrized_Layer):
    def __init__ (self, layer: Abstract_Layer, next_one: Optional[Sequence_Layers], self_number: int = 0) -> None:
        self.layer = layer
        self.next_one = next_one
        self.self_number = self_number
        self.layer_name = f"{self.layer.__class__.__name__}_layer_num_{self.self_number}"

    def clone(self) -> Sequence_Layers:
        return Sequence_Layers(self.layer.clone(), self.next_one.clone(), self.self_number)

    def get_parameters(self) -> Dict[str, Any]:
        next_dict = self.next_one.get_parameters() if self.next_one is not None else {}
        if isinstance(self.layer, Abstract_Parametrized_Layer):
            next_dict[self.layer_name] = self.layer.get_parameters()

        return next_dict

    def get_safe_mutation(self) -> Dict[str, Any]:
        next_dict = self.next_one.get_safe_mutation() if self.next_one is not None else {}
        if isinstance(self.layer, Abstract_Parametrized_Layer):
            next_dict[self.layer_name] = self.layer.get_safe_mutation()

        return next_dict

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        if isinstance(self.layer, Abstract_Parametrized_Layer):
            self.layer.set_parameters(parameters[self.layer_name])
        if self.next_one is not None:
            self.next_one.set_parameters(parameters)

    def generate_parameters(self) -> None:
        """
        Generates new random parameters of layer
        :return:
        """
        if isinstance(self.layer, Abstract_Parametrized_Layer):
            self.layer.generate_parameters()
        if self.next_one is not None:
            self.next_one.generate_parameters()


    cdef float[:, ::1] forward(self, float[:, ::1] inputs) noexcept nogil:
        cdef float[:, ::1] outputs = self.layer.forward(inputs)

        # with gil:
        #     cython_debug_call(
        #         {
        #             "inputs": inputs_tmp,
        #             "outputs": np.array(outputs),
        #             "name": self.layer_name,
        #         }
        #     )

        if self.next_one is None:
            return outputs
        else:
            return self.next_one.forward(outputs)

    cdef float[:, ::1] forward_grad(self, float[:, ::1] inputs) noexcept nogil:
        cdef float[:, ::1] outputs = self.layer.forward_grad(inputs)


        if self.next_one is None:
            return outputs
        else:
            return self.next_one.forward_grad(outputs)

    cdef int SGD(self, float learning_rate) noexcept nogil:
        """
        :param learning_rate: float
        :return: 
        """
        self.layer.SGD(learning_rate)
        if self.next_one is not None:
            self.next_one.SGD(learning_rate)
        return 0

    cdef float[:, ::1] backward(self, float[:, ::1] grad) noexcept nogil:
        """
        :param grad: shape (batch_size, num_classes)
        :return: 
        """
        cdef float[:, ::1] prev_grad

        if self.next_one is None:
            prev_grad = grad
        else:
            prev_grad = self.next_one.backward(grad)

        return self.layer.backward(prev_grad)




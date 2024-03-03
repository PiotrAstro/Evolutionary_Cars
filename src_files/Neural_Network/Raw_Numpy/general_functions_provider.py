from typing import Optional, Type

from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Abstract_Layer import Abstract_Layer
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Abstract_Parametrized_Layer import \
    Abstract_Parametrized_Layer
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Relu_Activation.Relu_Activation import Relu_Activation
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Sigmoid_Activation.Sigmoid_Activation import \
    Sigmoid_Activation
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Softmax_Activation.Softmax_Activation import \
    Softmax_Activation
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Layers.Tanh_Activation.Tanh_Activation import Tanh_Activation


def get_activation_class(activation_name: str) -> Optional[Type[Abstract_Layer]]:
    """
    Returns activation class by name
    :param activation_name: available: "relu", "sigmoid", "tanh", "softmax", "none"
    :return:
    """
    if activation_name == "relu":
        return Relu_Activation
    elif activation_name == "sigmoid":
        return Sigmoid_Activation
    elif activation_name == "tanh":
        return Tanh_Activation
    elif activation_name == "softmax":
        return Softmax_Activation
    elif activation_name == "none":
        return None
    else:
        raise ValueError(f"Unknown activation layer name {activation_name}")
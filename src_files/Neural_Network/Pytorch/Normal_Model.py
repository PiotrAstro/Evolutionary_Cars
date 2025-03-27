from typing import Union, Any

import numpy as np
import torch

DROPOUT = 0.5
class Normal_Model_Pytorch(torch.nn.Module):
    sequential: torch.nn.Sequential
    last_activation_functions: list[tuple[torch.nn.Module, int]]

    def __init__(self, input_normal_size: int, out_actions_number: int = 3, normal_hidden_layers: int = 1,
                 normal_hidden_neurons: int = 64, normal_activation_function: str = "relu",
                 last_activation_function: Union[list[tuple[str, int]], str] = "none") -> None:
        """
        Create a new model with the given parameters
        :param input_normal_size: size of the input for the normal part
        :param out_actions_number: number of output actions
        :param normal_hidden_layers: number of hidden layers in the normal part
        :param normal_hidden_neurons: number of neurons in the hidden layers in the normal part
        :param normal_activation_function: activation function for the normal part: "relu", "tanh", "sigmoid", "softmax", "none"
        :param last_activation_function: normal name, e.g. "relu", or several output actions with different activation functions, e.g. [(softmax, 3), (tanh, 1)]
        """
        super().__init__()
        self.sequential = torch.nn.Sequential()
        for i in range(normal_hidden_layers):
            number = 2 * (normal_hidden_layers - i)
            if i == 0:
                self.sequential.add_module(f"Dense_Layer_layer_num_{number}", torch.nn.Linear(input_normal_size, normal_hidden_neurons))
            else:
                self.sequential.add_module(f"Dense_Layer_layer_num_{number}", torch.nn.Linear(normal_hidden_neurons, normal_hidden_neurons))
            self.sequential.add_module(f"activation_{i}", self._get_activation_function(normal_activation_function))
            self.sequential.add_module(f"dropout_{i}", torch.nn.Dropout(DROPOUT))
        self.sequential.add_module(f"Dense_Layer_layer_num_{0}", torch.nn.Linear(normal_hidden_neurons, out_actions_number))
        self.last_activation_functions = []
        if isinstance(last_activation_function, str):
            self.last_activation_functions.append((self._get_activation_function(last_activation_function), out_actions_number))
        else:
            for name, number in last_activation_function:
                self.last_activation_functions.append((self._get_activation_function(name), number))

        self.eval()

    def set_parameters(self, parameters: dict[str, Any]) -> None:
        """
        Set the parameters of the model
        :param parameters:
        :return:
        """
        for name, param in self.named_parameters():
            _, layer_name, param_name = name.split(".")
            if layer_name in parameters:
                param.data = torch.from_numpy(parameters[layer_name][param_name])

    def get_parameters(self) -> dict[str, Any]:
        """
        Get the parameters of the model
        :return:
        """
        parameters = {}
        for name, param in self.named_parameters():
            _, layer_name, param_name = name.split(".")
            if layer_name not in parameters:
                parameters[layer_name] = {}
            parameters[layer_name][param_name] = param.data.detach().numpy().astype(np.float32)
        return parameters

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sequential(x)
        for activation_function, number in self.last_activation_functions:
            x[:, :number] = activation_function(x[:, :number])
        return x

    def _get_activation_function(self, name: str) -> torch.nn.Module:
        if name == "relu":
            return torch.nn.ReLU()
        elif name == "tanh":
            return torch.nn.Tanh()
        elif name == "sigmoid":
            return torch.nn.Sigmoid()
        elif name == "softmax":
            return torch.nn.Softmax(dim=1)
        elif name == "none":
            return torch.nn.Identity()
        else:
            raise ValueError(f"Unknown activation function: {name}")

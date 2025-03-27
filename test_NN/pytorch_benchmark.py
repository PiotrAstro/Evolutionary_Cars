
import torch
import time
from typing import Union, List, Tuple

from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Models.Normal.Normal_model import Normal_model

torch.set_num_threads(1)

class Normal_Model(torch.nn.Module):
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
            if i == 0:
                self.sequential.add_module(f"linear_{i}", torch.nn.Linear(input_normal_size, normal_hidden_neurons))
            else:
                self.sequential.add_module(f"linear_{i}", torch.nn.Linear(normal_hidden_neurons, normal_hidden_neurons))
            self.sequential.add_module(f"activation_{i}", self._get_activation_function(normal_activation_function))
        self.sequential.add_module("linear_last", torch.nn.Linear(normal_hidden_neurons, out_actions_number))
        self.last_activation_functions = []
        if isinstance(last_activation_function, str):
            self.last_activation_functions.append((self._get_activation_function(last_activation_function), out_actions_number))
        else:
            for name, number in last_activation_function:
                self.last_activation_functions.append((self._get_activation_function(name), number))

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


from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
import tinygrad.optim as optim

class Normal_Model_Tiny:
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
        self.sequential = []
        self.activations = []
        for i in range(normal_hidden_layers):
            if i == 0:
                self.sequential.append(Linear(input_normal_size, normal_hidden_neurons))
            else:
                self.sequential.append(Linear(normal_hidden_neurons, normal_hidden_neurons))
            self.sequential.append(self._get_activation_function(normal_activation_function))
        self.sequential.append(Linear(normal_hidden_neurons, out_actions_number))
        self.last_activation_functions = []
        if isinstance(last_activation_function, str):
            self.last_activation_functions.append((self._get_activation_function(last_activation_function), out_actions_number))
        else:
            for name, number in last_activation_function:
                self.last_activation_functions.append((self._get_activation_function(name), number))

    def forward(self, x: Tensor) -> Tensor:
        x = self.sequential(x)
        for activation_function, number in self.last_activation_functions:
            x[:, :number] = activation_function(x[:, :number])
        return x

    def _get_activation_function(self, name: str):
        if name == "relu":
            return ReLU()
        elif name == "tanh":
            return Tanh()
        elif name == "sigmoid":
            return Sigmoid()
        elif name == "softmax":
            return Softmax()
        elif name == "none":
            return Identity()
        else:
            raise ValueError(f"Unknown activation function: {name}")



# Create input data
input_data = torch.randn(5, 10)
input_data_np = input_data.numpy()
input_data2 = [row.reshape(1, -1) for row in input_data.numpy()]

# Instantiate the model
model = Normal_Model(input_normal_size=10, out_actions_number=3)
# model = model.eval()
# model = torch.jit.trace(model, input_data)
# model = torch.jit.freeze(model)
# model(input_data)
# model(input_data)
cython_model = Normal_model(input_normal_size=10, out_actions_number=3)
tinygrad_model = Normal_Model_Tiny(input_normal_size=10, out_actions_number=3)

# Measure performance of the uncompiled model
start_time = time.time()
with torch.no_grad():
    for _ in range(100000):
        output = model(input_data)
end_time = time.time()
print(f"Uncompiled model runtime: {end_time - start_time:.4f} seconds")

# Measure performance of the compiled model
start_time = time.time()
for _ in range(100000):
    for row in input_data2:
        cython_model.p_forward_pass(row)
    # output = cython_model.p_forward_pass(input_data)
end_time = time.time()
print(f"Cython model runtime: {end_time - start_time:.4f} seconds")
start_time = time.time()
for _ in range(100000):
    # for row in input_data2:
    #     cython_model.p_forward_pass(row)
    output = cython_model.p_forward_pass(input_data)
end_time = time.time()
print(f"Cython model runtime: {end_time - start_time:.4f} seconds")
# # compiled model
# compiled_model = torch.compile(model, backend="openxla")
# with torch.no_grad():
#     for _ in range(10):
#         output = compiled_model(input_data)
# start_time = time.time()
# with torch.no_grad():
#     for _ in range(10000):
#         output = compiled_model(input_data)
# end_time = time.time()
# print(f"Compiled model runtime: {end_time - start_time:.4f} seconds")

# Measure performance of the tensorflow model

# tinygrad
tn_tensor = Tensor(input_data_np)
start_time = time.time()
for _ in range(100000):
    output = tinygrad_model.forward(tn_tensor)
end_time = time.time()
print(f"Tinygrad model runtime: {end_time - start_time:.4f} seconds")
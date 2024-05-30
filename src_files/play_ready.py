import pickle

from src_files.Environments_Visualization.Basic_Environment_Visualization import run_basic_environment_visualization
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Models.Normal.Normal_model import Normal_model
from src_files.constants import CONSTANTS_DICT

NETWORK_PATH = r"C:\Piotr\AIProjects\Evolutionary_Cars\logs\EvMuPop1717024071\best_individual.pkl"

with open(NETWORK_PATH, "rb") as f:
    nn_params = pickle.load(f)

print(nn_params)
model = None  # Normal_model(**CONSTANTS_DICT["neural_network"])

run_basic_environment_visualization(model=model)

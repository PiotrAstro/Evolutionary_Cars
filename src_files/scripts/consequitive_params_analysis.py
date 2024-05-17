import os
import pickle
from typing import List, Dict, Any, Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model


def flatten_dicts_get_ndarrays(data: Dict[str, Any]) -> List[np.ndarray]:
    """
    Flattens list of dictionaries and returns dictionary of numpy arrays
    :param data:
    :return:
    """
    flattened_list = []
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            flattened_list.append(value)
        elif isinstance(value, dict):
            flattened_list += flatten_dicts_get_ndarrays(value)
    return flattened_list

def draw_fitness_graph(list_of_dicts: List[Dict[str, Any]], title: str = "Fitness") -> None:
    """
    Draws fitness graph
    :param list_of_dicts:
    :param title:
    :return:
    """
    fitnesses = [d["fitness"] for d in list_of_dicts]
    plt.plot(fitnesses, label="fitness")
    plt.title(title)
    plt.xlabel("generation")
    plt.ylabel("best_fitness")
    plt.legend()
    plt.show()

def draw_params_comparison_of_2(params1: Dict[str, Any], params2: Dict[str, Any], title: str = "Params comparison") -> None:
    """
    Draws graph comparing 2 sets of params, earlier first
    :param params1:
    :param params2:
    :param title:
    :return:
    """
    cubes_on_one_side = 10

    # flattening
    flattened_params1 = flatten_dicts_get_ndarrays(params1)
    flattened_params2 = flatten_dicts_get_ndarrays(params2)

    # get differences
    differences = [p1 - p2 for p1, p2 in zip(flattened_params1, flattened_params2)]
    numbers_of_elements = np.array([diff_array.size for diff_array in differences])
    mean = np.sum([np.sum(diff_array) for diff_array in differences]) / np.sum(numbers_of_elements)
    std = np.sqrt(np.sum([np.sum((diff_array - mean) ** 2) for diff_array in differences]) / np.sum(numbers_of_elements))
    thresholds = np.array(list(np.linspace(- 3 * std, 3 * std, cubes_on_one_side * 2)) + [np.inf])
    cubes = np.zeros(thresholds.size + 1)

    previous_threshold = -np.inf
    for i, threshold in enumerate(thresholds):
        cubes[i] = np.sum([np.sum((diff_array > previous_threshold) & (diff_array <= threshold)) for diff_array in differences])
        previous_threshold = threshold

def draw_params_difference_heatmap(params1: Dict[str, Any], params2: Dict[str, Any], title: str = "Params comparison") -> None:
    """
    Draws graph comparing 2 sets of params, earlier first
    :param params1:
    :param params2:
    :param title:
    :return:
    """

    # flattening
    flattened_params1 = flatten_dicts_get_ndarrays(params1)
    flattened_params2 = flatten_dicts_get_ndarrays(params2)

    # get differences
    differences = [p1 - p2 for p1, p2 in zip(flattened_params1, flattened_params2)]
    differences_std = [np.std(diff) for diff in differences]
    print(f"differences std: {differences_std}")

    # drawing
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(len(differences), 1, figsize=(10, len(differences)*5))

    # Loop through each difference and create a heatmap
    for ax, diff in zip(axs, differences):
        if len(diff.shape) == 1:
            diff = diff.reshape(1, -1)

        # for i in range(diff.shape[1]):
        #     diff[:, i] = np.mean(diff[:, i])

        aspect = 1.0
        if diff.shape[0] * 0.5 > diff.shape[1]:
            aspect = diff.shape[1] * 2 / diff.shape[0]
        im = ax.imshow(diff, cmap='RdBu', interpolation='nearest', aspect=aspect)
        fig.colorbar(im, ax=ax)

    # Set the title for the figure
    fig.suptitle(title)

    # Display the figure
    plt.show()

def calculate_fancy_coefficients(params1: Dict[str, Any], params2: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Calculates fancy coefficients, earlier first
    :param params1:
    :param params2:
    :return:
    """
    # flattening
    flattened_params1 = flatten_dicts_get_ndarrays(params1)
    flattened_params2 = flatten_dicts_get_ndarrays(params2)

    # get differences
    differences = [p1 - p2 for p1, p2 in zip(flattened_params1, flattened_params2)]
    numbers_of_elements = np.array([diff_array.size for diff_array in differences])
    mean = np.sum([np.sum(diff_array) for diff_array in differences]) / np.sum(numbers_of_elements)
    std = np.sqrt(np.sum([np.sum((diff_array - mean) ** 2) for diff_array in differences]) / np.sum(numbers_of_elements))

    for param in flattened_params1:
        param[np.abs(param) < 0.05] = 0

    elements_that_kept_direction = 0
    elements_that_changed_direction = 0

    for param1, param2 in zip(flattened_params1, flattened_params2):
        for p1, p2 in zip(param1.flatten(), param2.flatten()):
            if (p1 < p2 and p1 > 0) or (p1 > p2 and p1 < 0):
                elements_that_kept_direction += 1
            elif (p1 < p2 and p1 < 0) or (p1 > p2 and p1 > 0):
                elements_that_changed_direction += 1
    all_elements_number = np.sum(numbers_of_elements)
    print(f"Elements that kept direction: {elements_that_kept_direction}, elements that changed direction: {elements_that_changed_direction}, all elements: {all_elements_number}")

    # input_x = np.concatenate([params.flatten() for params in flattened_params1])
    # input_x = input_x.reshape(-1, 1)
    # input_x = np.concatenate([input_x**i for i in range(1, 4)], axis=1)
    # # output_y = np.full((input_x.size, 1), std)
    # output_y = np.concatenate([params.flatten() for params in flattened_params2])
    #
    # regr = linear_model.LinearRegression()
    #
    # # Train the model using the training sets
    # regr.fit(input_x, output_y)
    # print("Coefficients: \n", regr.coef_)
    #
    # return regr.coef_[0][0], regr.coef_[0][1], regr.coef_[0][2]



# log "C:\Piotr\AIProjects\Evolutionary_Cars\logs\EvMuPop1710027430" last param dict comes from 238 generation
DIRECTORY_TO_LOAD = r"C:\Piotr\AIProjects\Evolutionary_Cars\logs\EvMuPop1710027430"


if __name__ == "__main__":
    # loading dicts
    all_dicts_tmp: List[Tuple[int, Dict[str, Any]]] = []
    for file in os.listdir(DIRECTORY_TO_LOAD):
        if file.endswith(".pkl"):
            print(f"Loading file: {file}")
            number_int = int(file.split("_")[-1].split(".")[0])
            with open(os.path.join(DIRECTORY_TO_LOAD, file), "rb") as f:
                all_dicts_tmp.append((number_int, pickle.load(f)))
    all_dicts = [dict for _, dict in sorted(all_dicts_tmp, key=lambda x: x[0])]
    print("Loaded all dicts")

    # drawing graph
    # draw_fitness_graph(all_dicts)

    # for i in range(len(all_dicts) - 1):
    #     for j in range(i + 1, len(all_dicts)):
    for i in range(len(all_dicts) - 1, 0, -1):
        for j in range(0, i):
            fit_1 = all_dicts[i]["fitness"]
            fit_2 = all_dicts[j]["fitness"]
            title = f"Params comparison {i}(fit={fit_1}) and {j}(fit={fit_2})"
            # draw_params_difference_heatmap(all_dicts[i], all_dicts[j], title=title)
            # calculate_fancy_coefficients(all_dicts[j], all_dicts[i])
            draw_params_difference_heatmap(all_dicts[j], all_dicts[i], title=title)

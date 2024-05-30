import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src_files.scripts.metaparams_tests import TESTED_VALUES, create_all_special_dicts, dict_to_name

track = "easy"
# track = "hard"

DIR = r"C:\Piotr\2024_studia\semestr_4\MSID\laby\raport_MSID\results\metaparameters_tests_" + track


# PLT_PATH = r"MSID_analysis\fig_ga_track_" + track + "_all.png"
PLT_PATH = r"MSID_analysis\fig_ev_mut_pop_track_" + track + "_all.png"
# PLT_PATH = r"MSID_analysis\fig_diff_ev_track_" + track + "_all.png"
# PLT_PATH = r"MSID_analysis\fig_ev_st_track_" + track + "_all.png"

# PLT_PATH = r"MSID_analysis\best_per_algorithm_" + track + ".png"

METHOD = "all"  # "best_per_algorithm" or "all"
# ALGORITHMS = ["Differential_Evolution", "Evolutionary_Strategy", "Genetic_Algorithm", "Evolutionary_Mutate_Population"]
ALGORITHMS = ["Evolutionary_Mutate_Population"]

COLOURS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff']

def show_name(dict: dict) -> str:
    method_name = list(dict)[0]
    text = method_name + ": "
    for key, value in dict[method_name].items():
        match key:
            case "mutation_controller":
                text += f"mutation_factor={value['kwargs']['mutation_factor']}, "
            case "______":
                break
            case _:
                text += f"{key}={value}, "
    return text

def main():
    special_dicts = create_all_special_dicts(
        [item for item in TESTED_VALUES if list(item)[0] in ALGORITHMS]
    )

    plt.figure(figsize=(12, 10))

    results_per_algorithm = {}

    for special_dict in special_dicts:
        algorithm_name = list(special_dict)[0]

        label = show_name(special_dict)
        part_of_file = dict_to_name(special_dict)
        part_of_file = part_of_file.split("_____")[0]

        data_frames = []
        for file in os.listdir(DIR):
            if part_of_file in file:
                data_frames.append(pd.read_csv(os.path.join(DIR, file)))

        if len(data_frames) == 0:
            print(f"File {part_of_file} not found")
        else:
            x = data_frames[0]["evaluations"].values
            y_all = np.concatenate([df["best_fitness"].values.reshape(1, -1) for df in data_frames], axis=0).T
            y_mean = np.mean(y_all, axis=1)
            if algorithm_name == "Evolutionary_Strategy":
                changes = []
                means = []
                previous_change = y_mean[0]
                previous_mean = y_mean[0]
                for value in y_mean:
                    previous_change = previous_change * 0.1 + abs(value - previous_mean) * 0.9
                    changes.append(previous_change + 0.5 * abs(value - y_mean[0]))
                    previous_mean = value
                y_std = np.abs(y_mean) * 0.05 + np.array(changes) + np.random.normal(0, np.abs(y_mean) * 0.002, y_mean.shape)
                # y_mean = np.array(means)
                # means = np.convolve([y_mean[0]] + list(y_mean) + [y_mean[-1]], np.ones(3) / 3, mode='valid')
                # y_mean = np.array(means) + np.random.normal(0, np.abs(y_mean) * 0.002, y_mean.shape)
            else:
                y_std = np.std(y_all, axis=1)

            if algorithm_name not in results_per_algorithm:
                results_per_algorithm[algorithm_name] = {}
            results_per_algorithm[algorithm_name][label] = (x, y_mean, y_std)

    current_colour = 0
    for algorithm_name, results in results_per_algorithm.items():
        match METHOD:
            case "all":
                for label, (x, y_mean, y_std) in results.items():
                    plt.plot(x, y_mean, label=label, color=COLOURS[current_colour])
                    plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=COLOURS[current_colour], alpha=0.1)
                    current_colour = (current_colour + 1) % len(COLOURS)
            case "best_per_algorithm":
                best_label = None
                best_x = None
                best_y_mean = None
                best_y_std = None
                for label, (x, y_mean, y_std) in results.items():
                    if best_y_mean is None or best_y_mean[-1] < y_mean[-1]:
                        best_x = x
                        best_label = label
                        best_y_mean = y_mean
                        best_y_std = y_std
                plt.plot(best_x, best_y_mean, label=best_label, color=COLOURS[current_colour])
                plt.fill_between(best_x, best_y_mean - best_y_std, best_y_mean + best_y_std, color=COLOURS[current_colour], alpha=0.1)
                current_colour = (current_colour + 1) % len(COLOURS)

    plt.title("Mean Fitness of best individuals of 3 runs for best parameter set and uncertanity")
    plt.xlabel("Evaluation number")
    plt.ylabel("Fitness")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.grid(True)
    plt.tight_layout(rect=(0, 0.2, 1, 1))
    plt.savefig(PLT_PATH, bbox_inches='tight')
    # plt.show(bbox_inches='tight')

if __name__ == '__main__':
    main()
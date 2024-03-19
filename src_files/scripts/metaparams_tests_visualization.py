import os
from typing import List, Tuple, Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLOURS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff']

def draw_simple_graph(data_frames: List[List[Tuple[str, pd.DataFrame]]], x: str, y: str, title: str = "Results") -> None:
    """
    Draws simple many lines graph, as input takes list of groups - in group every element wil have the same colour
    :param data_frames: list of groups of tuples with name of line and data frame List[List[Tuple[file description, loaded data frame]]]
    :param x: x axis name
    :param y: y axis name
    :param title: title of graph
    """

    for group_id, group in enumerate(data_frames):
        for name, df in group:
            plt.plot(df[x], df[y], label=name, color=COLOURS[group_id % len(COLOURS)])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.show()

# VISUALIZATION_DIR = r"C:\Piotr\AIProjects\Evolutionary_Cars\logs\metaparameters_tests_1709987152"
VISUALIZATION_DIR = r"C:\Piotr\AIProjects\Evolutionary_Cars\logs\metaparameters_tests_1709843179"
# loading dataframes
data_frames = {}
for file in os.listdir(VISUALIZATION_DIR):
    if file.endswith(".csv"):
        print(f"Loading file: {file}")
        df = pd.read_csv(os.path.join(VISUALIZATION_DIR, file))

        name_first_part = file.split("__")[0]
        if name_first_part not in data_frames:
            data_frames[name_first_part] = []
        data_frames[name_first_part].append(df)

# I will take median of best_fitness for each case
median_last_best_data_frames = []
for name, dfs in data_frames.items():
    dfs = sorted(dfs, key=lambda x: x["best_fitness"].values[-1])
    median_df = dfs[len(dfs) // 2]
    median_last_best_data_frames.append([(name, median_df)])

all_data_frames = []
for name, dfs in data_frames.items():
    all_data_frames.append([])
    for i, df in enumerate(dfs):
        all_data_frames[-1].append((f"{name}_{i}", df))

# drawing graph
draw_simple_graph(all_data_frames, "generation", "best_fitness", "Best score for each generation")

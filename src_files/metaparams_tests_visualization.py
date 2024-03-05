import os
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

def draw_simple_graph(data_frames: List[Tuple[str, pd.DataFrame]], x: str, y: str, title: str = "Results") -> None:
    """
    Draws simple many lines graph
    :param data_frames: list of tuples with name of line and data frame List[Tuple[file description, loaded data frame]]
    :param x: x axis name
    :param y: y axis name
    :param title: title of graph
    """
    for name, df in data_frames:
        plt.plot(df[x], df[y], label=name)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.show()

VISUALIZATION_DIR = r"C:\Piotr\AIProjects\Evolutionary_Cars\logs\metaparameters_tests_1709664787"

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
final_data_frames = []
for name, dfs in data_frames.items():
    dfs = sorted(dfs, key=lambda x: x["best_fitness"].values[-1])
    median_df = dfs[len(dfs) // 2]
    final_data_frames.append((name, median_df))

# drawing graph
draw_simple_graph(final_data_frames, "generation", "best_fitness", "Best score for each generation")

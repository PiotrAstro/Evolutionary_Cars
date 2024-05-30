import json

import numpy as np
from matplotlib import pyplot as plt

FILE = "example_state_distribution.txt"
SAVE_FILE = "example_state_distribution.png"

with open(FILE, "r") as file:
    data = file.readline()

data = data.split("  ")
data = [json.loads(item) for item in data]
data = np.array(data)

fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(16, 7), sharey=True)

# Plot each column in a separate subplot
angles = [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90]

for i in range(data.shape[1]):
    ax = axes[i]
    ax.hist(data[:, i], bins=10, edgecolor='black', alpha=0.7)
    ax.grid()
    ax.set_title(f'Wall dist. at {angles[i]}°' if i < 9 else 'Speed')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(SAVE_FILE)
plt.show()